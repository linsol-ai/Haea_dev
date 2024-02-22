from pydantic import BaseModel, Field
from typing import List


class ModelConfig(BaseModel):
    """A config specification of model."""

    num_heads: int = Field(ge=5, default=10)

    n_encoder_layers: int = Field(ge=1, default=3)

    n_decoder_layers: int = Field(ge=1, default=3)

    dropout: float = Field(ge=0, default=0.1)


class TrainingConfig(BaseModel):
    """A config specification of training the DVAE."""

    learning_rate: float = 2e-4
    """The learning rate."""

    warmup_step: int = Field(ge=0, default=4000)

    batch_size: int = Field(ge=0, default=256)
    """"The batch size."""

    max_epochs: int = Field(ge=0, default=100)
    """The maximum number of training epochs."""

    gradient_clip_val: float | None = None
    """The value to clip the gradients to."""

    tgt_time_len: int = Field(ge=0, default=7)

    train_offset: int = 0

    air_variable: List[str]

    surface_variable: List[str]

    only_input_variable: List[str] = []

    constant_variable: List[str] = []

    

class TrainingRunConfig(BaseModel):
    """A config specification of the training run."""

    model: ModelConfig
    """A config specification of the model."""

    training: TrainingConfig
    """A config specification of the training."""

    seed: int = 123
    """The random seed."""
