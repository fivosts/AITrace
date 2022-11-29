import os
import sys
import pathlib
import typing

from absl import app, flags

from models import models
from samplers import samplers
from proto import aitrace_pb2
from util import memory
from util import pbutil

from eupy.native import logger as l
from eupy.hermes import client

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "level",
  20,
  "Define logging level of logger"
)

flags.DEFINE_string(
  "notify_me",
  None,
  "Set receiver mail address to notify for program failures or termination."
)

flags.DEFINE_integer(
  "notify_me_level",
  5,
  "Define logging level of mail client"
)

flags.DEFINE_boolean(
  "color", True, "Colorize or not, logging messages"
)

flags.DEFINE_boolean(
  "step", False, "Enable step execution on debug logs (debug level must be selected)"
)

flags.DEFINE_string(
  "stop_after",
  None,
  'Stop application early. Valid options are: "dataset", or "train".',
)

flags.DEFINE_string(
  "workspace_dir",
  "/tmp/AITrace",
  "Root path of the working space directory. Corpus, dataset, model and all meta files"
  "will be stored here. Default value is /tmp folder.",
)

flags.DEFINE_boolean(
  "monitor_mem_usage",
  False,
  "Plot application's RAM and GPU memory usage."
)

flags.DEFINE_string(
  "config",
  "",
  "Path to an ai_trace.Instance proto file."
)

class Instance(object):
  """
  Generic instance encapsulating machine learning pipeline (datasets, model and sampler).
  """

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> "Instance":
    return cls(pbutil.FromFile(path, aitrace_pb2.Instance()))

  def __init__(self, config: aitrace_pb2.Instance):
    """
    Initialize an instance.

    Args:
      config: An Instance protobuf.
    """
    self.working_dir = None
    self.model       = None
    self.sampler     = None
    self.config      = config

    if config.HasField("working_dir"):
      self.working_dir = pathlib.Path(os.path.join(FLAGS.workspace_dir, config.working_dir))
      self.working_dir = self.working_dir.expanduser().resolve()

    if config.HasField("model"):
      self.model = models.Model(config.model, self.working_dir)
    if config.HasField("sampler"):
      self.sampler = samplers.Sampler(config.sampler, self.working_dir)

  def Create(self) -> None:
    """Create datasets and model."""
    self.model.Create()
    return

  def Train(self) -> None:
    self.model.Train()
    return

  def Sample(self) -> None:
    if self.sampler:
      if not self.sampler.is_server_sampling:
        self.model.Sample(self.sampler)
      else:
        self.model.ServerSample(self.sampler)
    return

def ConfigFromFlags() -> aitrace_pb2.Instance:
  """
  Parse model config file into Instance object.
  """
  config_path = pathlib.Path(FLAGS.config)
  if not config_path.is_file():
    raise FileNotFoundError (f"AITrace --config file not found: '{config_path}'")
  return pbutil.FromFile(config_path, aitrace_pb2.Instance())

def RunApplication(instance: Instance) -> None:
  """
  Run application according to the configuration and flags provided.

  Args:
    instance: The model instance to act on.
  """
  if FLAGS.stop_after not in {None, "dataset", "train"}:
    raise ValueError(f"Invalid --stop_after argument: '{FLAGS.stop_after}'")
  if instance.model:
    instance.Create()
    if FLAGS.stop_after == "dataset":
      return
    instance.Train()
    if FLAGS.stop_after == "train":
      return
    instance.Sample()
  return

def main():
  """
  Main application.
  """
  RunApplication(
    instance = Instance(ConfigFromFlags()),
  )
  return

def bootstrap(*args, **kwargs):
  """
  Bootstrap function for application.
  Performs early app initialization.
  """
  mail = None
  if FLAGS.notify_me:
    mail = client.initClient(FLAGS.notify_me)

  l.initLogger(
    name = "AITrace",
    lvl = FLAGS.level,
    mail = (mail, FLAGS.notify_me_level),
    colorize = FLAGS.color,
    step = FLAGS.step
  )
  if FLAGS.monitor_mem_usage:
    mem_monitor_threads = memory.init_mem_monitors(
      pathlib.Path(FLAGS.workspace_dir).resolve()
    )
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(1)
  except NotImplementedError as e:
    raise e
    sys.exit(1)
  except Exception as e:
    l.getLogger().error(e)
    if mail:
      mail.send_message("AITrace", e)
    raise e
    sys.exit(1)

  if mail:
    mail.send_message(
      "AITrace",
      "Program terminated successfully at {}".format(
        datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S")
      )
    )
  sys.exit(0)

if __name__ == "__main__":
  """
  App entrypoint
  """
  app.run(bootstrap)
