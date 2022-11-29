import typing
import numpy as np

from util import pytorch

from eupy.native import logger as l
from absl import flags

torch = pytorch.torch
FLAGS = flags.FLAGS

PAD_IDX = 4095

def ConfigModelParams(config) -> 'AIVinci':
  """
  Construct model pipeline from configuration.
  """
  return AIVinci(config)

class EmbeddingLayer(torch.nn.Module):
  """
  Input Embedding layer between raw inputs and sequential input.
  """
  def __init__(self, config):
    super().__init__()
    self.word_embeddings = torch.nn.Embedding(
      num_embeddings = 4096,
      embedding_dim = config.embedding_size,
      padding_idx = PAD_IDX
    )
    self.layer_norm = torch.nn.LayerNorm(
      config.embedding_size,
      eps = config.layer_norm_eps
    )
    self.dropout = torch.nn.Dropout(config.dropout_prob)
    return

  def forward(self, input_ids):
    ## (Batch, Sequence_length, Sequence size) -> (Batch, Sequence_length, Sequence size, Embedding size)
    embeddings = self.word_embeddings(input_ids)
    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    """
    https://gist.github.com/williamFalcon/f27c7b90e34b4ba88ced042d9ef33edd
    """
    return embeddings

class Sequential(torch.nn.Module):
  """
  LSTM encoder for input data.
  """
  def __init__(self, config):
    super().__init__()
    self.LSTM = torch.nn.LSTM(
      input_size  = config.embedding_size * 2,
      hidden_size = config.hidden_size,
      num_layers  = config.num_hidden_layers,
      batch_first = True,
      dropout     = config.dropout_prob
    )
    return

  def forward(self, input_ids, input_lengths):
    flatten_embeds = input_ids.view(input_ids.size(0), input_ids.size(1), -1)
    input_packed = torch.nn.utils.rnn.pack_padded_sequence(
      flatten_embeds.to('cpu'),
      input_lengths.to('cpu').squeeze(-1),
      batch_first = True,
      enforce_sorted = False,
    )
    out, hidden = self.LSTM(flatten_embeds)
    last_timestep = torch.stack([out[b,l-1,:] for b, l in zip(range(out.size(0)), input_lengths.squeeze(-1))])
    return out, last_timestep

class VisitingLabelHead(torch.nn.Module):
  """
  Prediction head for museum visiting styles.
  """
  def __init__(self, config):
    super().__init__()
    # self.summarizer = torch.nn.Conv1d(
    #   in_channels = 147,
    #   out_channels = 1,
    #   kernel_size = 3,
    #   stride = 1,
    #   padding = 1
    # )
    self.linear  = torch.nn.Linear(config.hidden_size, config.output_size, bias = True)
    self.dropout = torch.nn.Dropout(p = config.dropout_prob)
    # self.l2 = torch.nn.Linear(config.intermediate_size, config.output_size, bias = True)
    self.output  = torch.nn.Linear(config.output_size, 5, bias = True)
    self.softmax = torch.nn.Softmax(dim = 1)
    return

  def calculate_loss(self, output_labels, target_label) -> typing.Tuple["torch.Tensor", typing.Tuple[int, int]]:

    if target_label is None:
      return None, (None, None)

    ## Calculate categorical label loss.
    loss_fn = torch.nn.CrossEntropyLoss()
    label_loss = loss_fn(output_labels.to(torch.float32), target_label.squeeze(1))

    ## Calculate top-1 accuracy of predictions across batch.
    hits, total = 0, int(output_labels.size(0))
    probs       = self.softmax(output_labels)
    outputs     = torch.argmax(probs, dim = 1)
    for out, target in zip(outputs, target_label):
      if out == target:
        hits += 1
    return label_loss, (hits, total)

  def forward(self, output_representations, target_label = None):
    # summarized = self.summarizer(output_representations)
    o1 = self.linear(output_representations.squeeze(1))
    # o2 = self.l2(o1)
    d1 = self.dropout(o1)
    o3 = self.output(d1)
    return o3, self.calculate_loss(o3, target_label)

class NextStepHead(torch.nn.Module):
  """
  Prediction head for museum visitor's next step.
  """
  def __init__(self, config):
    super().__init__()
    self.linear = torch.nn.Linear(config.hidden_size, 2*config.output_size, bias = True)
    self.dropout = torch.nn.Dropout(p = config.dropout_prob)
    # self.l2 = torch.nn.Linear(config.hidden_size, config.output_size, bias = True)
    self.exhibit = torch.nn.Linear(2*config.output_size, 45, bias = True)
    self.time    = torch.nn.Linear(2*config.output_size, 1, bias = True)

    self.softmax = torch.nn.Softmax(dim = 1)
    self.scaler  = lambda x: x / 50
    self.reverse_scaler = lambda x: x * 50
    return

  def calculate_loss(self, output_id, output_time, target_id) -> typing.Tuple["torch.Tensor", typing.Tuple[int, int, int, int]]:

    if target_id is None:
      return None, None, (None, None, None)

    ## Calculate categorical loss of next exhibit id.
    loss_fn1 = torch.nn.CrossEntropyLoss()
    exh_id_loss = loss_fn1(output_id.to(torch.float32), target_id[:,0])

    ## Calculate Mean-Squared-Error loss for next attendance time.
    loss_fn2 = torch.nn.MSELoss()
    time_att_loss = loss_fn2(self.scaler(output_time.to(torch.float32).squeeze(1)), self.scaler(target_id[:,1].to(torch.float32)))

    ## Calculate top-1, top-3 and top-5 prediction accuracy of next exhibit id.
    hits_1, hits_3, hits_5, total = 0, 0, 0, int(output_id.size(0))
    probs       = self.softmax(output_id)
    outputs_1   = torch.topk(probs, 1, dim = 1).indices
    outputs_3   = torch.topk(probs, 3, dim = 1).indices
    outputs_5   = torch.topk(probs, 5, dim = 1).indices
    for out1, out3, out5, target in zip(outputs_1, outputs_3, outputs_5, target_id[:,0]):
      if target in out1:
        hits_1 += 1
        hits_3 += 1
        hits_5 += 1
      elif target in out3:
        hits_3 += 1
        hits_5 += 1
      elif target in out5:
        hits_5 += 1
    return exh_id_loss, time_att_loss, (hits_1, hits_3, hits_5, total) # + next_id_loss

  def forward(self, output_representations, target_id = None):
    o1 = self.linear(output_representations)
    d1 = self.dropout(o1)

    exh, att = self.exhibit(d1), self.time(d1)
    # o3 = self.output(o1)
    return exh, att, self.calculate_loss(exh, att, target_id)

class AIVinci(torch.nn.Module):
  """
  Base model pipeline.
  """
  def __init__(self, config):
    super().__init__()
    self.input_embeds = EmbeddingLayer(config)
    self.encoder      = Sequential(config)
    self.label_head   = VisitingLabelHead(config)
    self.next_id_head = NextStepHead(config)
    return

  def forward(self,
              input_ids     : torch.Tensor,
              input_lengths : torch.Tensor,
              target_id     : torch.Tensor = None,
              target_label  : torch.Tensor = None,
              temperature   : int          = None,
              ) -> "torch.Tensor":

    if target_id is None and target_label is None:
      sampling = True
      if len(input_ids) > 1:
        raise ValueError("Currently only batch size of 1 is supported for sampling.")
    else:
      sampling = False

    input_embeddings   = self.input_embeds(input_ids)
    encoded, last_step = self.encoder(input_embeddings, input_lengths)

    label_out, (label_loss, label_accuracy) = self.label_head(last_step, target_label)
    next_id, next_attendance_time, (next_id_loss, next_time_loss, next_id_accuracy) = self.next_id_head(last_step, target_id)

    if sampling:
      if temperature:
        sampled_id = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature = temperature,
            logits = next_id,
            validate_args = False if "1.9." in torch.__version__ else None,
          ).sample()
        sampled_label = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature = temperature,
            logits = label_out,
            validate_args = False if "1.9." in torch.__version__ else None,
          ).sample()
      else:
        sampled_id    = torch.nn.softmax(next_id, dim = -1)
        sampled_label = torch.nn.softmax(label_out, dim = -1)

      next_id   = torch.argmax(sampled_id,    dim = -1)
      label_out = torch.argmax(sampled_label, dim = -1)
      total_loss = None
    else:
      total_loss = label_loss + next_id_loss + next_time_loss

    return {
      'total_loss'          : total_loss,
      'label_loss'          : label_loss,
      'next_id_loss'        : next_id_loss,
      'next_time_loss'      : next_time_loss,
      'label_accuracy'      : label_accuracy,
      'next_id_accuracy'    : next_id_accuracy,
      'label_out'           : label_out,
      'next_id'             : next_id,
      'next_attendance_time': next_attendance_time,
      'sample'              : input_ids,
    }
