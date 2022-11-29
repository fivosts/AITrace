import pathlib
import typing
import multiprocessing
import os
import queue
import threading

from backend import server
from models import builders
from models.torch_lstm import torch_lstm
from datasets import datasets
from datasets import encoders
from samplers import samplers
from samplers import samples_database
from proto import model_pb2
from proto import internal_pb2
from util import crypto
from util import pbutil
from util import commit

from absl import flags

from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "num_train_steps",
  None,
  "Bypass num_train_steps provided by protobuf file."
)

flags.DEFINE_integer(
  "num_samples",
  None,
  "Specify number of samples to produce."
)

class Model(object):
  """
  Abstract representation of current AITrace model.
  """
  def __init__(self, config: model_pb2.Model, base_dir: pathlib.Path):
    # Error early, so that a cache isn't created.
    if not isinstance(config, model_pb2.Model):
      t = type(config).__name__
      raise TypeError(f"Config must be a Model proto. Received: '{t}'")

    self.config = model_pb2.Model()
    # Validate config options.
    self.config.CopyFrom(builders.AssertIsBuildable(config))
    if FLAGS.num_train_steps:
      self.config.training.num_train_steps = FLAGS.num_train_steps
      
    self.dataset = datasets.Dataset(config.dataset, base_dir)
    self.hash    = self._ComputeHash(self.dataset, self.config)

    # Create the necessary cache directories.
    self.cache_path = base_dir / "model" / self.hash
    self.cache_path.mkdir(exist_ok = True, parents = True)
    (self.cache_path / "checkpoints").mkdir(exist_ok=True)
    (self.cache_path / "samples"    ).mkdir(exist_ok=True)

    self._created = False

    # Create symlink to encoded corpus.
    symlink = self.cache_path / "corpus"
    if not symlink.is_symlink():
      os.symlink(
        os.path.relpath(
          pathlib.Path(self.dataset.encoded.url[len("sqlite:///") :]).parent,
          self.cache_path,
        ),
        symlink,
      )

    # Validate metadata against cache.
    if (self.cache_path / "META.pbtxt").exists():
      cached_meta = pbutil.FromFile(
        pathlib.Path(self.cache_path / "META.pbtxt"), internal_pb2.ModelMeta()
      )
      config_to_compare = model_pb2.Model()
      config_to_compare.CopyFrom(self.config)
      # config_to_compare.dataset.ClearField("contentfiles")
      # These fields should have already been cleared, but we'll do it again
      # so that metadata comparisons don't fail when the cached meta schema
      # is updated.
      cached_to_compare = model_pb2.Model()
      cached_to_compare.CopyFrom(cached_meta.config)
      # cached_to_compare.dataset.ClearField("contentfiles")
      if config_to_compare != cached_to_compare:
        raise SystemError("Metadata mismatch: {} \n\n {}".format(config_to_compare, cached_to_compare))
      self.meta = cached_meta
    else:
      self.meta = internal_pb2.ModelMeta()
      self.meta.config.CopyFrom(self.config)
      self._WriteMetafile()

    ## Store current commit
    commit.saveCommit(self.cache_path)

    self.backend = {
      model_pb2.NetworkArchitecture.TORCH_LSTM: torch_lstm.torchLSTM,
    }[config.architecture.backend](self.config, self.cache_path, self.hash)
    return

  def Create(self) -> None:
    """
    Create dataset for model and backend for training.
    """
    if self._created:
      return False
    self._created = True
    self.dataset.Create()
    self.backend.Create()

  def Train(self) -> None:
    """
    Initialize model training.
    """
    self.backend.Train(self.dataset)
    return

  def Sample(self, sampler: samplers.Sampler) -> None:
    """
    Initialize model sampling.
    """
    self.backend.InitSampling(sampler)
    samples_db = samples_database.SamplesDatabase("sqlite:///{}".format(sampler.cache_path / sampler.sample_db_name), must_exist = False)
    nsamples = 0
    with samples_db.Session(commit = True) as ses:
      try:
        while True:
          label, sequence = self.backend.Sample(sampler)
          nsamples += sampler.batch_size
          # for label, sequence in zip(labels, sequences):
          label = encoders.ids_to_labels[label]
          print("=== AITRACE VISITOR ===")
          print()
          print("== LABEL: {}".format(label))
          print()
          print("== VISITED IDS - ATTENDANCE TIME ==")
          print()
          print("\n".join(["{}\t{}".format(i, t) for i, t in sequence]))
          print()
          sha256 = crypto.sha256_str(label + "\n".join(["{},{}".format(i, t) for i, t in sequence]))
          exists = ses.query(samples_database.Sample).filter_by(sha256 = sha256).first()
          if not exists:
            entry = samples_database.Sample.FromArgs(
              id       = samples_db.count,
              sha256   = sha256,
              label    = label,
              exhibits = sequence,
            )
            ses.add(entry)
            ses.commit()
          ## Add to database here as well.
          if FLAGS.num_samples and nsamples >= FLAGS.num_samples:
            ses.commit()
            break
      except KeyboardInterrupt:
        ses.commit()
    l.getLogger().info("Finished sampling, generating {} unique samples".format(samples_db.count))
    return

  def ServerSample(self, sampler: samplers.Sampler) -> None:
    """
    Start socket server, collect sampler inputs, and serve back.
    """
    try:
      in_queue  = queue.Queue()
      out_queue = queue.Queue()
      sample_method = multiprocessing.Value('i', True if sampler.prediction_type == "step" else False)
      temperature   = multiprocessing.Value('d', sampler.temperature)
      sthread   = threading.Thread(
        target = server.serve,
        kwargs = {
          'in_queue'  : in_queue,
          'out_queue' : out_queue,
          'port'      : sampler.server_port,
          'sample_method': sample_method,
          'temperature'  : temperature,
        },
        daemon = True
      )
      sthread.start()
      for entry in sampler.ExpectInputs(in_queue, num_inputs = -1):
        sampler.input_feed  = entry['input_feed']
        sampler.temperature = temperature.value
        sampler.prediction_type = {
          True: "step",
          False: "full",
        }[sample_method.value]
        label, sequence    = self.backend.Sample(sampler)
        entry['visitor_label'] = encoders.ids_to_labels[label]
        if sampler.prediction_type == "step":
          entry['next_id']              = int(sequence[-1][0])
          entry['next_attendance_time'] = int(sequence[-1][-1])
        else:
          entry['predicted_visit'] = [[int(x[0]), int(x[1])] for idx, x in enumerate(sequence) if idx >= len(sampler.input_feed)]
        sampler.PublishOutputs(entry, out_queue)
        sampler.prediction_type = sampler.config.prediction_type
        sampler.temperature     = sampler.config.temperature_micros / 1e6
    except KeyboardInterrupt:
      l.getLogger().info("Wrapping up sampling server...")
    except Exception as e:
      raise e
    return

  @staticmethod
  def _ComputeHash(corpus_ : datasets.Dataset,
                   config  : model_pb2.Model,
                   ) -> str:
    """Compute model hash.

    The hash is computed from the ID of the corpus and the serialized
    representation of the config proto. The number of epochs that the model is
    trained for does not affect the hash, since we can share checkpoints
    between different models if the only variable is the epoch count. E.g.
    we have a model trained for 10 epochs, we can use the checkpoint as the
    starting point for a training a model for 20 epochs.

    Args:
      corpus: A corpus instance.
      config: A Model config proto.

    Returns:
      The unique model ID.
    """
    config_to_hash = model_pb2.Model()
    config_to_hash.CopyFrom(config)
    config_to_hash.ClearField("dataset")
    hash_list = [corpus_.hash, config_to_hash.SerializeToString()]
    return crypto.sha1_list(hash_list)

  def _WriteMetafile(self) -> None:
    pbutil.ToFile(self.meta, self.cache_path / "META.pbtxt")
