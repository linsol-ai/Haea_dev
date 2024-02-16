from pydantic import BaseModel, Field


class LinearSchedulerConfig(BaseModel):
    """A config specification of the parameter scheduler."""

    start: float = 1.0
    """The initial value of the parameter."""

    end: float = 1.0
    """The final value of the parameter."""

    warmup: float = Field(ge=0, le=1, default=0.0)
    """Fraction of the total training steps for the warmup phase."""

    cooldown: float = Field(ge=0, le=1, default=0.0)
    """Fraction of the total training steps for the cooldown phase."""


class ExponentialSchedulerConfig(BaseModel):
    """A config specification of the parameter scheduler."""

    start: float = 1.0
    """The initial value of the parameter."""

    min: float = 0.5
    """The min value of the parameter."""

    anneal_rate: float = 1e-6
    """The anneal_rate of the parameter."""

class DVAEModelConfig(BaseModel):
    """A config specification of the DVAE model."""

    codebook_size: int = Field(ge=0, default=256)
    """The number of vectors in the codebook."""

    codebook_vector_dim: int = Field(ge=0, default=256)
    """The dimension of the codebook vector."""

    hidden_dim: int = Field(ge=0, default=256)
    """The dimension of the hidden vector."""

    channels: int = Field(ge=0)
    """The number of channels in the input image."""

    num_layers: int = 3
    """The number of layers"""

    num_resnet_blocks: int = 2
    """The number of resnet blocks"""


class DVAETrainingConfig(BaseModel):
    """A config specification of training the DVAE."""

    train_variable = variable 

    learning_rate: float = 5e-4
    """The learning rate."""
    lr_decay_rate: float = 0.98

    """The minimum factor value for the learning rate scheduler."""
    kl_div_weight_scheduler: LinearSchedulerConfig = Field(
        default_factory=lambda: LinearSchedulerConfig(start=0, end=1e-4, warmup=0.1, cooldown=0.2)
    )
    """A config specification of the KL Divergence weight scheduler."""

    temperature_scheduler: ExponentialSchedulerConfig = Field(
        default_factory=lambda: ExponentialSchedulerConfig(
            start=1., min=0.5, anneal_rate=1e-6 
        )
    )
    """A config specification of the temperature scheduler."""

    batch_size: int = Field(ge=0, default=256)
    """"The batch size."""

    max_epochs: int = Field(ge=0, default=100)
    """The maximum number of training epochs."""

    gradient_clip_val: float | None = None
    """The value to clip the gradients to."""

    save_vis_every_n_step: int = Field(ge=0, default=100)
    """Save the validation visualizations every n epochs."""

    num_vis: int = Field(ge=0, default=10)
    """The number of validation visualizations to save every `save_vis_every_n_epochs` steps."""


class TrainingRunConfig(BaseModel):
    """A config specification of the training run."""

    model: DVAEModelConfig
    """A config specification of the model."""

    training: DVAETrainingConfig
    """A config specification of the training."""

    seed: int = 123
    """The random seed."""
