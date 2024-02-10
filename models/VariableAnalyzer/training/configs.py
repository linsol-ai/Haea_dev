from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """A config specification of model."""

    num_heads: int = Field(ge=5, default=10)

    n_encoder_layers: int = Field(ge=1, default=3)

    n_decoder_layers: int = Field(ge=1, default=3)

    dropout: float = Field(ge=0, default=0.1)


class DVAETrainingConfig(BaseModel):
    """A config specification of training the DVAE."""

    learning_rate: float = 2e-4
    """The learning rate."""

    warmup_step: float = Field(ge=1, default=4000)

    weight_decay: float = 

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
