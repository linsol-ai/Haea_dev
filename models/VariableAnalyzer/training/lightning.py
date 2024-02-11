import pytorch_lightning as pl
import torch.optim.optimizer
from torch.optim import AdamW
import torch.nn.functional as F

from models.VariableAnalyzer.models.model import VariableAnalyzer
from models.VariableAnalyzer.training.configs import TrainingConfig
from models.VariableAnalyzer.training.params_schedule import CosineWarmupScheduler

class TrainModule(pl.LightningModule):

    def __init__(self, *, model: VariableAnalyzer, var_len: int, predict_dim: int, max_iters: int, config: TrainingConfig | None = None):
        super().__init__()
        self.var_len = var_len
        self.max_iters = max_iters
        self.predict_dim = predict_dim
        self.model = model
        self.config = TrainingConfig() if config is None else config
        self.save_hyperparameters(self.config.dict(), ignore=["model", "config"])


    def configure_optimizers(self) -> AdamW:  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(
            optimizer,
            gamma= self.config.lr_decay_rate
        )
        return [optimizer], [scheduler]


    def _step(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        src = batch[:, :self.var_len]
        tgt = batch[:, self.var_len:]
        output = self.model(src, tgt)
        loss = F.mse_loss(tgt[:, :, :self.predict_dim], output[:, :, :self.predict_dim])
        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")
        return loss

    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, _: int) -> None:  # noqa: D102
        self._step(batch, "test")
