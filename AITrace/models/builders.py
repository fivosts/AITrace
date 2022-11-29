"""This file builds Keras models from Model config protos."""

from absl import flags
from proto import model_pb2
from util import pbutil

FLAGS = flags.FLAGS


def AssertIsBuildable(config: model_pb2.Model) -> model_pb2.Model:
  """Assert that a model configuration is buildable.

  Args:
    config: A model proto.

  Returns:
    The input model proto, unmodified.

  Raises:
    UserError: If the model is not buildable.
    InternalError: If the value of the training.optimizer field is not
      understood.
  """
  # Any change to the Model proto schema will require a change to this function.
  try:
    pbutil.AssertFieldIsSet(config, "dataset")
    pbutil.AssertFieldIsSet(config, "architecture")
    pbutil.AssertFieldIsSet(config, "training")
    pbutil.AssertFieldIsSet(config.architecture, "backend")
    if config.architecture.backend == model_pb2.NetworkArchitecture.TORCH_LSTM:
      pbutil.AssertFieldConstraint(
        config.architecture,
        "embedding_size",
        lambda x: 0 < x,
        "NetworkArchitecture.embedding_size must be > 0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "num_hidden_layers",
        lambda x: 0 < x,
        "NetworkArchitecture.num_hidden_layers must be > 0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "dropout_prob",
        lambda x: 0 <= x <= 1.0,
        "NetworkArchitecture.dropout_prob "
        "must be >= 0.0 and <= 1.0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "hidden_size",
        lambda x: 0 < x,
        "TrainingOptions.hidden_size must be > 0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "intermediate_size",
        lambda x: 0 < x,
        "TrainingOptions.intermediate_size must be > 0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "output_size",
        lambda x: 0 < x,
        "TrainingOptions.output_size must be > 0",
      )
      pbutil.AssertFieldConstraint(
        config.architecture,
        "layer_norm_eps",
        lambda x: 0 < x,
        "NetworkArchitecture.layer_norm_eps "
        "must be > 0.0",
      )
    pbutil.AssertFieldConstraint(
      config.training,
      "num_epochs",
      lambda x: 1 <= x,
      "TrainingOptions.num_epochs must be >= 1",
    )
    pbutil.AssertFieldConstraint(
      config.training,
      "num_warmup_steps",
      lambda x: 0 <= x,
      "TrainingOptions.num_warmup_steps must be >= 0",
    )
    pbutil.AssertFieldIsSet(
      config.training,
      "random_seed",
    )
    pbutil.AssertFieldConstraint(
      config.training,
      "batch_size",
      lambda x: 0 < x,
      "TrainingOptions.batch_size must be > 0",
    )
    pbutil.AssertFieldIsSet(config.training, "optimizer")
    if config.training.HasField("adam_optimizer"):
      pbutil.AssertFieldConstraint(
        config.training.adam_optimizer,
        "initial_learning_rate_micros",
        lambda x: 0 <= x,
        "AdamOptimizer.initial_learning_rate_micros must be >= 0",
      )
    elif config.training.HasField("rmsprop_optimizer"):
      pbutil.AssertFieldConstraint(
        config.training.rmsprop_optimizer,
        "initial_learning_rate_micros",
        lambda x: 0 <= x,
        "RmsPropOptimizer.initial_learning_rate_micros must be >= 0",
      )
      pbutil.AssertFieldConstraint(
        config.training.rmsprop_optimizer,
        "learning_rate_decay_per_epoch_micros",
        lambda x: 0 <= x,
        "RmsPropOptimizer.learning_rate_decay_per_epoch_micros must be >= 0",
      )
    else:
      raise SystemError(
        "Unrecognized value: 'TrainingOptions.optimizer'"
      )
  except Exception as e:
    raise e
  return config
