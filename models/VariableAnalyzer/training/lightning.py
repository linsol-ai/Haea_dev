import pytorch_lightning as pl
import torch.optim.optimizer
from torch.optim import AdamW
from einops import rearrange
from models.model import VariableAnalyzer
from training.configs import TrainingConfig
from training.params_schedule import CosineWarmupScheduler
from typing import List, Tuple

class TrainModule(pl.LightningModule):
    """A PyTorch Lightning training module for the `DVAE`."""

    def __init__(self, *, model: VariableAnalyzer, config: TrainingConfig | None = None):
        """Init the DVAE training module.

        Args:
            dvae: An instance of the Discrete Variational Auto-Encoder to be used.
            config: The DVAE's training configuration. If `None`, default configuration
                will be used.
        """
        super().__init__()
        self.model = model
        self.config = TrainingConfig() if config is None else config
        self.save_hyperparameters(self.config.dict(), ignore=["model", "config"])


    def configure_optimizers(self) -> tuple[list[AdamW], list[CosineWarmupScheduler]]:  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.config.warmup_step, max_iters=self.config.max_iters)
        return [optimizer], [scheduler]


    def _step(self, batch: List[Tuple[torch.Tensor, torch.Tensor]], mode: str) -> torch.Tensor:
        sr
        

        self.log(f"{mode}/loss", loss, prog_bar=mode == "train")
        self.log("temperature", self._temperature_scheduler.get_value())
        self.log("kl_div_weight", self._kl_div_weight_scheduler.get_value())
        
        return loss

    def training_step(self, batch: List[Tuple[torch.Tensor, torch.Tensor]], _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def on_train_epoch_end(self) -> None:  # noqa: D102
        self._kl_div_weight_scheduler.step()

    def validation_step(self, batch: List[Tuple[torch.Tensor, torch.Tensor]], _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: List[Tuple[torch.Tensor, torch.Tensor]], _: int) -> None:  # noqa: D102
        self._step(batch, "test")
