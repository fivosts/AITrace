import typing
import numpy as np
import pathlib
import progressbar
import tqdm

from datasets import datasets
from datasets import encoders
from models import backends
from samplers import samplers
from models.torch_lstm import model
from models.torch_lstm import optimizer
from models.torch_lstm import data_generator
from models.torch_lstm import hooks

from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "only_sample",
  False,
  "Select to skip training and sample only."
)

class torchLSTM(backends.BackendBase):
  def __init__(self, config, path, model_hash):
    super(torchLSTM, self).__init__(config, path, model_hash)
    return

  def Create(self) -> None:
    """
    Set up model.
    """
    self.num_epochs       = self.config.training.num_epochs
    self.num_warmup_steps = self.config.training.num_warmup_steps
    self.validation_split = self.config.training.validation_percentage
    self.train_batch_size = self.config.training.batch_size
    self.learning_rate    = self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6

    seed = np.random.RandomState().randint(0, 2**32-1)
    self.torch.manual_seed(seed)
    self.torch.cuda.manual_seed_all(seed)
    return

  def model_step(self,
                 input_ids,
                 input_lengths,
                 target_id,
                 target_label,
                 ) -> typing.Dict[str, "torch.Tensor"]:
    """
    Run model step for batch and return loss.
    """
    outputs = self.model(
      input_ids     = input_ids,
      input_lengths = input_lengths,
      target_id     = target_id,
      target_label  = target_label,
    )
    return outputs

  def sample_model_step(self,
                        input_ids,
                        input_lengths,
                        temperature = None,
                        ) -> typing.Dict[str, "torch.Tensor"]:
    """
    Run forward function in sampling mode.
    """
    outputs = self.model(
      input_ids     = input_ids,
      input_lengths = input_lengths,
      temperature   = temperature
    )
    return outputs

  def Train(self, dataset_instance: datasets.Dataset, **kwargs) -> None:
    """
    Train LSTM model.
    """
    ## Define training dataset and dataloader.
    dataset = dataset_instance.getEncodedData()
    try:
      num_splits = int(1 // self.validation_split)
    except ZeroDivisionError:
      num_splits = 1
    split_size = int(len(dataset) * self.validation_split)

    for split_idx in range(num_splits):
      # raise NotImplementedError("Refactor this")
      train_dataset = dataset[:split_size * split_idx] + dataset[split_size * (split_idx + 1):]
      validation_dataset = dataset[split_size * split_idx: split_size * (split_idx + 1)]
      assert len(train_dataset) + len(validation_dataset) == len(dataset), "Mismatch between train, validation and total dataset length.".format(len(train_dataset), len(validation_dataset), len(dataset))

      ## Initialize the model
      self.model = model.ConfigModelParams(self.config.architecture).to(self.pytorch.device)

      ## Define torch datasets
      train_set        = data_generator.UnrolledDataset(train_dataset)
      val_unrolled_set = data_generator.UnrolledDataset(validation_dataset)
      val_set          = data_generator.Dataset(validation_dataset)

      ## Define dataloaders.
      train_loader = self.torch.utils.data.dataloader.DataLoader(
        dataset     = train_set,
        batch_size  = self.train_batch_size,
        sampler     = self.torch.utils.data.RandomSampler(train_set, replacement = False),
        num_workers = 0,
        drop_last   = False,
      )
      if len(validation_dataset) > 0:
        val_unrolled_loader = self.torch.utils.data.dataloader.DataLoader(
          dataset     = val_unrolled_set,
          batch_size  = self.train_batch_size,
          sampler     = self.torch.utils.data.RandomSampler(val_unrolled_set, replacement = False),
          num_workers = 0,
          drop_last   = False,
        )
        val_loader = self.torch.utils.data.dataloader.DataLoader(
          dataset     = val_set,
          batch_size  = self.train_batch_size,
          sampler     = self.torch.utils.data.RandomSampler(val_set, replacement = False),
          num_workers = 0,
          drop_last   = False,
        )

      ## Also create scheduler and optmizer.
      opt, scheduler = optimizer.create_optimizer_and_scheduler(
        model           = self.model,
        num_train_steps = (self.num_epochs * len(train_set)) // self.train_batch_size,
        warmup_steps    = self.num_warmup_steps,
        learning_rate   = self.learning_rate,
      )
      ## Load checkpoint, if exists.
      current_step = self.loadCheckpoint(self.model, opt, scheduler, split_idx)

      ## Setup train logging hook.
      train_hook = hooks.tensorMonitorHook(
        self.logfile_path / "split_{}".format(split_idx),
        current_step,
        step_freq = 500,
      )
      if FLAGS.only_sample:
        return
      for ep in tqdm.auto.trange(self.num_epochs, desc="Epoch", leave = False):
        if current_step > ep * len(train_loader):
          continue
        self.model.train()
        for batch in tqdm.tqdm(train_loader, total = len(train_loader), desc="Batch", leave = False):
          outputs = self.model_step(
            input_ids     = batch['input_ids'].to(self.pytorch.device),
            input_lengths = batch['input_lengths'].to(self.pytorch.device),
            target_id     = batch['target_id'].to(self.pytorch.device),
            target_label  = batch['target_label'].to(self.pytorch.device),
          )
          loss = outputs['total_loss'].mean()
          loss.backward()
          opt.step()
          scheduler.step()
          train_hook.step(
            label_loss       = outputs['label_loss'].mean().item(),
            label_accuracy   = outputs['label_accuracy'][0] / outputs['label_accuracy'][1],
            next_id_loss     = outputs['next_id_loss'].mean().item(),
            next_time_loss   = outputs['next_time_loss'].mean().item(),
            next_id_accuracy = (outputs['next_id_accuracy'][0] / outputs['next_id_accuracy'][-1], outputs['next_id_accuracy'][1] / outputs['next_id_accuracy'][-1], outputs['next_id_accuracy'][2] / outputs['next_id_accuracy'][-1]),
            total_loss       = loss.item(),
            learning_rate    = scheduler.get_last_lr()[0],
            # batch_execution_time_ms = exec_time_ms,
            # time_per_sample_ms      = exec_time_ms / self.train_batch_size,
          )
          current_step += 1
        self.saveCheckpoint(self.model, opt, scheduler, split_idx, current_step)
        if len(validation_dataset) > 0:
          val_outputs = self.Validate(unrolled = val_unrolled_loader, standard = val_loader)
          train_hook.end_epoch(**val_outputs)
    return

  def Validate(self, **dataloaders: typing.Dict[str, "torch.nn.DataLoader"]) -> typing.Dict[str, float]:
    """
    Validate model, return loss and accuracy.
    """
    self.model.eval()
    outputs = {}
    for name, dataloader in dataloaders.items():
      outputs = {
        'val_total_loss_{}'.format(name)     : 0,
        'val_label_loss_{}'.format(name)     : 0,
        'val_next_time_loss_{}'.format(name) : 0,
        'val_next_id_loss_{}'.format(name)   : 0,
        'val_next_id_acc_{}'.format(name)    : [0, 0, 0],
      }
      for batch in tqdm.tqdm(dataloader, total = len(dataloader), desc="Validation", leave = False):
        out = self.model_step(
          input_ids     = batch['input_ids'].to(self.pytorch.device),
          input_lengths = batch['input_lengths'].to(self.pytorch.device),
          target_id     = batch['target_id'].to(self.pytorch.device),
          target_label  = batch['target_label'].to(self.pytorch.device),
        )
        outputs["val_total_loss_{}".format(name)]     += out['total_loss'].mean().item()
        outputs["val_label_loss_{}".format(name)]     += out['label_loss'].mean().item()
        outputs["val_next_time_loss_{}".format(name)] += out['next_time_loss'].mean().item()
        outputs["val_next_id_loss_{}".format(name)]   += out['next_id_loss'].mean().item()
        outputs["val_next_id_acc_{}".format(name)][0] += out['next_id_accuracy'][0] / out['next_id_accuracy'][-1]
        outputs["val_next_id_acc_{}".format(name)][1] += out['next_id_accuracy'][1] / out['next_id_accuracy'][-1]
        outputs["val_next_id_acc_{}".format(name)][2] += out['next_id_accuracy'][2] / out['next_id_accuracy'][-1]
      outputs["val_total_loss_{}".format(name)]     /= len(dataloader)
      outputs["val_label_loss_{}".format(name)]     /= len(dataloader)
      outputs["val_next_time_loss_{}".format(name)] /= len(dataloader)
      outputs["val_next_id_loss_{}".format(name)]   /= len(dataloader)
      outputs["val_next_id_acc_{}".format(name)][0] /= len(dataloader)
      outputs["val_next_id_acc_{}".format(name)][1] /= len(dataloader)
      outputs["val_next_id_acc_{}".format(name)][2] /= len(dataloader)
    return outputs

  def InitSampling(self, sampler: samplers.Sampler, split_idx: int = 0, **kwargs) -> None:
    self.model = model.ConfigModelParams(self.config.architecture).to(self.pytorch.device)
    current_step = self.loadCheckpoint(self.model, None, None, split_idx)
    self.model.eval()
    return

  def Sample(self, sampler: samplers.Sampler, **kwargs) -> None:
    """
    Sample a trained instance of the model.
    """
    batch = data_generator.compute_sample_batch(sampler.input_feed, sampler.batch_size)
    outputs = self.sample_model_step(
      input_ids     = batch['input_ids'].to(self.pytorch.device),
      input_lengths = batch['input_lengths'].to(self.pytorch.device),
      temperature   = sampler.temperature,
    )
    if sampler.prediction_type == "full":
      while outputs['next_id'].item() != encoders.terminals['leaves']:
        if outputs['next_id'].item() != encoders.terminals['enters']:
          batch = data_generator.increment_sample_batch(
            [outputs['next_id'].item(), outputs['next_attendance_time'].item()],
            [x for x in batch['input_ids'][0].cpu().numpy()],
            sampler.batch_size
          )
        outputs = self.sample_model_step(
          input_ids     = batch['input_ids'].to(self.pytorch.device),
          input_lengths = batch['input_lengths'].to(self.pytorch.device),
          temperature   = sampler.temperature,
        )

    if outputs['next_id'].item() != encoders.terminals['enters']:
      batch = data_generator.increment_sample_batch(
        [outputs['next_id'].item(), outputs['next_attendance_time'].item()],
        [x for x in batch['input_ids'][0].cpu().numpy()],
        sampler.batch_size
      )
    else:
      return self.Sample(sampler)
    return (outputs['label_out'].cpu().item(), batch['input_ids'].squeeze(0).cpu().numpy())

  def saveCheckpoint(self, model: model.AIVinci, opt, scheduler, split_idx: int, current_step: int):
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    ckpt_comp = lambda x: self.ckpt_path / "{}-{}-{}.pt".format(x, split_idx, current_step)

    if isinstance(model, self.torch.nn.DataParallel):
      self.torch.save(model.module.state_dict(), ckpt_comp("model"))
    else:
      self.torch.save(model.state_dict(), ckpt_comp("model"))
    self.torch.save(opt.state_dict(), ckpt_comp("optimizer"))
    self.torch.save(scheduler.state_dict(), ckpt_comp("scheduler"))
    with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
      mf.write("split_idx: {}, train_step: {}\n".format(split_idx, current_step))
    return

  def loadCheckpoint(self, model: model.AIVinci, opt, scheduler, split_idx: int) -> int:
    """
    Load model checkpoint. Loads either most recent epoch, or selected checkpoint through FLAGS.
    """
    if not (self.ckpt_path / "checkpoint.meta").exists():
      return -1

    with open(self.ckpt_path / "checkpoint.meta", 'r') as mf:
      get_step  = lambda x: int(x.replace("\n", "").replace("split_idx: ", "").split("train_step: ")[1])
      entries   = set({get_step(x) for x in mf.readlines() if x})

    ckpt_step = max(entries)
    ckpt_comp = lambda x: self.ckpt_path / "{}-{}-{}.pt".format(x, split_idx, ckpt_step)

    if isinstance(model, self.torch.nn.DataParallel):
        model.module.load_state_dict(self.torch.load(ckpt_comp("model")))
    else:
      try:
        model.load_state_dict(self.torch.load(ckpt_comp("model"), map_location=self.pytorch.device))
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    if opt is not None and scheduler is not None and ckpt_step > 0:
      opt.load_state_dict(self.torch.load(ckpt_comp("optimizer"), map_location=self.pytorch.device))
      scheduler.load_state_dict(self.torch.load(ckpt_comp("scheduler"), map_location=self.pytorch.device))
    model.eval()
    return ckpt_step
