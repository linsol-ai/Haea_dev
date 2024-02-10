from pydantic import BaseModel, Field

class LinearSchedulerConfig(BaseModel):
    """A config specification of the parameter scheduler."""

    """The final value of the parameter."""

    warmup: float = Field(ge=0, le=1, default=0.0)
    """Fraction of the total training steps for the warmup phase."""

    warmup_step: int = Field(ge=0, le=1, default=0.0)


class ModelConfig(BaseModel):
    """A config specification of model."""

    num_heads: int = Field(ge=0, default=10)

    n_encoder_layers: int = Field(ge=0, default=3)

    n_decoder_layers: int = Field(ge=0, default=3)

    dropout: float = Field(ge=0, default=0.1)


class DVAETrainingConfig(BaseModel):
    """A config specification of training the DVAE."""

    learning_rate: float = 2e-4
    """The learning rate."""

    lr_rate_warm_up

    lr_decay_rate: float = 0.98

    batch_size: int = Field(ge=0, default=256)
    """"The batch size."""

    max_epochs: int = Field(ge=0, default=100)
    """The maximum number of training epochs."""

    gradient_clip_val: float | None = None
    """The value to clip the gradients to."""



class TrainingRunConfig(BaseModel):
    """A config specification of the training run."""

    model: DVAEModelConfig
    """A config specification of the model."""

    training: DVAETrainingConfig
    """A config specification of the training."""

    seed: int = 123
    """The random seed."""
