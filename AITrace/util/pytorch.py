"""A wrapper module to include tensorflow with some options"""
from absl import flags
import os

from util import gpu

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "pt_cpu_only",
  False,
  "Do not use GPU/TPU in pytorch session."
)

import torch

offset_device = None
devices       = None
device        = None
num_gpus      = None

def initPytorch() -> None:
  global offset_device
  global devices
  global device
  global num_gpus
  if FLAGS.pt_cpu_only:
    device = torch.device("cpu")
    num_gpus = 0
  else:
    # if num_gpus is > 1 we'll use nn.DataParallel.
    # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
    # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
    # trigger an error that a device index is missing. Index 0 takes into account the
    # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
    # will use the first GPU in that env, i.e. GPU#1
    offset_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      available_gpus = gpu.getGPUID()
      devices = ["cuda:{}".format(str(x['id'])) for x in available_gpus]
    # device         = torch.device(
    #   "cuda:{}".format(str(available_gpus[0]['id'])) if torch.cuda.is_available() and available_gpus else "cpu"
    # )
    device = "cuda:0"
    num_gpus = torch.cuda.device_count()
    # if device.type == "cuda":
    torch.cuda.set_device("cuda:0")
  return