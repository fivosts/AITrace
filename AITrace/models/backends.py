"""
AITrace neural network backends.
"""
import typing
import pathlib
import numpy as np
from absl import flags

from util import pytorch

FLAGS = flags.FLAGS

class BackendBase(object):
  """The base class for a language model backend.

  A language model backend encapsulates all of the neural network logic.
  """
  def __init__(
    self,
    config,
    fs_cache,
    hash: str,
  ):
    self.config       = config
    self.cache        = fs_cache
    self.ckpt_path    = self.cache / "checkpoints"
    self.logfile_path = self.cache / "logs"
    self.logfile_path.mkdir(exist_ok = True, parents = True)
    self.hash         = hash

    self.pytorch   = pytorch
    self.torch     = pytorch.torch
    self.pytorch.initPytorch()


    self.is_validated      = False
    self.trained           = False
    return

  def Create() -> None:
    """Initialize model architecture"""
    raise NotImplementedError

  def Train(self, dataset: "Dataset", **extra_kwargs) -> None:
    """Train the backend."""
    raise NotImplementedError

  def Sample(self, sampler, **kwargs) -> typing.Tuple[int, typing.List[typing.Tuple[int, float]]]:
    """
    Sample the model based on sampler specs.

    Returns a tuple that contains:
      A) The predicted label of the visitor,
      B) List of tuples, (next_exhibit_id, next_exhibit_time)
    """
    raise NotImplementedError
