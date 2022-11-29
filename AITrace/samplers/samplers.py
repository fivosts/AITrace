import pathlib
import os
import json
import typing
import queue

from proto import sampler_pb2
from proto import internal_pb2
from util import crypto
from util import pbutil
from util import commit

from absl import flags
from eupy.native import logger as l

def AssertConfigIsValid(config: sampler_pb2.Sampler) -> sampler_pb2.Sampler:
  """Assert that a sampler configuration contains no invalid values.

  Args:
    config: A sampler configuration proto.

  Returns:
    The sampler configuration proto.

  Raises:
    UserError: If there are configuration errors.
  """
  try:
    if ((not config.HasField("train_set"))
      and (not config.HasField("validation_set"))
      and (not config.HasField("input_feed"))
      and (not config.HasField("server_sampling"))):
      raise ValueError(config)
    pbutil.AssertFieldConstraint(
      config, "batch_size", lambda x: 0 < x, "Sampler.batch_size must be > 0"
    )
    pbutil.AssertFieldConstraint(
      config,
      "prediction_type",
      lambda x: x in {"step", "full"},
      "sampler.prediction_type can only be full or step."
    )
    pbutil.AssertFieldConstraint(
      config,
      "temperature_micros",
      lambda x: 0 < x,
      "Sampler.temperature_micros must be > 0",
    )
    return config
  except pbutil.ProtoValueError as e:
    raise ValueError(e)

class Sampler(object):
  """
  Abstract representation of current AITrace model.
  """
  @property
  def is_server_sampling(self):
    return self.config.HasField("server_sampling")
  
  def __init__(self,
               config         : sampler_pb2.Sampler,
               base_dir       : pathlib.Path,
               sample_db_name : str = "samples.db"
               ):
    """Instantiate a sampler.

    Args:
      config: A Sampler message.

    Raises:
      TypeError: If the config argument is not a Sampler proto.
      UserError: If the config contains invalid values.
    """
    if not isinstance(config, sampler_pb2.Sampler):
      t = type(config).__name__
      raise TypeError(f"Config must be a Sampler proto. Received: '{t}'")

    self.config = sampler_pb2.Sampler()
    self.config.CopyFrom(AssertConfigIsValid(config))
    self.hash = self._ComputeHash(self.config)

    self.temperature     = self.config.temperature_micros / 1e6
    self.batch_size      = self.config.batch_size
    self.sample_db_name  = sample_db_name
    self.prediction_type = self.config.prediction_type
    if self.is_server_sampling:
      pbutil.AssertFieldIsSet(self.config, "server_port")
      self.server_port = self.config.server_port

    # Create the necessary cache directories.
    self.cache_path = base_dir / "samples" / self.hash
    self.samples_directory = self.cache_path / "samples"
    self.cache_path.mkdir(exist_ok = True, parents = True)

    if self.config.HasField("input_feed"):
      self.input_feed = [[0, 0]]

    meta = internal_pb2.SamplerMeta()
    meta.config.CopyFrom(self.config)
    pbutil.ToFile(meta, path = self.cache_path / "META.pbtxt")
    commit.saveCommit(self.cache_path)
    return

  def symlinkModelDB(self,
                     db_path   : pathlib.Path,
                     model_hash: int,
                     ) -> None:
    """
    Create symbolic link entry in sampler workspace. In one 
    model's workspace, there is one sampler.db for each different
    sampler. Each sampler holds a directory of all models it has 
    sampled with symbolic links created in this function.
    """
    assert os.path.isdir(db_path), "Parent path of database is not an existing path!"
    (self.samples_directory / model_hash).mkdir(exist_ok = True)

    for file in db_path.iterdir():
      symlink = self.samples_directory / model_hash / file.name
      if not symlink.is_symlink():
        os.symlink(
          os.path.relpath(
            db_path / file.name,
            self.samples_directory / model_hash
          ),
          symlink
        )
    return

  def ExpectInputs(self,
                   queue: queue.Queue,
                   num_inputs: int = -1,
                   ) -> 'json':
    """
    Busy waiting inputs in the multithreaded input queue.
    """
    yielded_els = 0
    while num_inputs == -1 or yielded_els < num_inputs:
      cur = queue.get(block = True)
      yielded_els += 1
      yield json.loads(cur)

  def PublishOutputs(self,
                     jsonf: typing.Dict,
                     out_queue: queue.Queue
                     ) -> None:
    """
    Serve predictions back to the input.
    """
    out_queue.put(bytes(json.dumps(jsonf), encoding="utf-8"), block = True)
    return

  @staticmethod
  def _ComputeHash(config: sampler_pb2.Sampler) -> str:
    """Compute sampler hash.

    The hash is computed from the serialized representation of the config
    proto.
    """
    return crypto.sha1(config.SerializeToString())

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Sampler):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)
