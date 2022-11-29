import typing
import pickle
import numpy as np
import pathlib
import progressbar

from datasets import datasets
from models import backends
from models.torch_lstm import model
from models.torch_lstm import data_generator
from util import pytorch

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS
torch = pytorch.torch

def compute_sample_batch(input_feed: typing.List[typing.Tuple[int, int]], batch_size: int) -> typing.Dict[str, torch.LongTensor]:
  """
  Receive samplers input feed and return prepared input tensors.
  input_feed is a list of tuples (exh_id, attendance_time)
  """
  return {
    'input_ids'    : torch.LongTensor(input_feed).unsqueeze(0).repeat(batch_size, 1, 1),
    'input_lengths': torch.LongTensor([len(input_feed)]).unsqueeze(0).repeat(batch_size, 1),
  }

def increment_sample_batch(next_steps: typing.Tuple[int, int], input_feed: typing.List[typing.Tuple[int, int]], batch_size: int) -> typing.Dict[str, torch.LongTensor]:
  """
  Increment sample batch input with previously predicted step.
  """
  nid, natt = next_steps
  input_feed.append([nid, natt])
  return compute_sample_batch(input_feed, batch_size)

class Dataset(torch.utils.data.Dataset):
  """
  Torch dataloder without unrolling.
  """
  def __init__(self, dataset: datasets.Dataset) -> None:
    """
    Dataset constructor.
    """
    self.dataset = dataset
    # self.path    = checkpoint
    self.corpus  = self.compute_dataset()
    return

  def __len__(self):
    return len(self.corpus)

  def __getitem__(self, idx: int):
    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx
    return self.corpus[idx]

  def compute_dataset(self) -> typing.List[typing.Dict[str, torch.Tensor]]:
    """
    Preprocess encoded data to unrolled torch tensors.

    Let's use
      A) 12 bits to represent the delta timestamp
      B) 6 bits to represent exhibits.
    """
    corpus = []
    max_seq_len = 0
    for seq, label in self.dataset:
      seq   = [[int(x) for x in ent.split(',')] for ent in seq.split('\n')]
      max_seq_len = max(max_seq_len, len(seq) - 1)
    for seq, label in self.dataset:
      seq = [[int(x) for x in ent.split(',')] for ent in seq.split('\n')]
      corpus.append({
        'input_ids'    : torch.LongTensor(seq[:len(seq)-1] + [[4095, 4095]]*(max_seq_len - len(seq) + 1)),
        'input_lengths': torch.LongTensor([len(seq)-1]),
        'target_id'    : torch.LongTensor(seq[len(seq)-1]),
        'target_label' : torch.LongTensor([int(label)]),
      })
    np.random.shuffle(corpus)
    return corpus

class UnrolledDataset(torch.utils.data.Dataset):
  """
  Torch dataloder for Hecht Training Data.
  """
  def __init__(self, dataset: datasets.Dataset) -> None:
    """
    Dataset constructor.
    """
    self.dataset = dataset
    # self.path    = checkpoint
    self.corpus  = self.compute_dataset()
    return

  def __len__(self):
    return len(self.corpus)

  def __getitem__(self, idx: int):
    if idx < 0:
      if -idx > len(self):
        raise ValueError("absolute value of index should not exceed dataset length")
      idx = len(self) + idx
    return self.corpus[idx]

  def compute_dataset(self) -> typing.List[typing.Dict[str, torch.Tensor]]:
    """
    Preprocess encoded data to unrolled torch tensors.

    Let's use
      A) 12 bits to represent the delta timestamp
      B) 6 bits to represent exhibits.
    """
    corpus = []
    max_seq_len = 0
    for seq, label in self.dataset:
      seq   = [[int(x) for x in ent.split(',')] for ent in seq.split('\n')]
      max_seq_len = max(max_seq_len, len(seq) - 1)
    for seq, label in self.dataset:
      seq   = [[int(x) for x in ent.split(',')] for ent in seq.split('\n')]
      label = int(label)
      ridx  = 1
      while ridx < len(seq):
        corpus.append({
          'input_ids'    : torch.LongTensor(seq[:ridx] + [[4095, 4095]]*(max_seq_len - ridx)),
          'input_lengths': torch.LongTensor([ridx]),
          'target_id'    : torch.LongTensor(seq[ridx]),
          'target_label' : torch.LongTensor([label]),
        })
        ridx += 1
    np.random.shuffle(corpus)
    return corpus
