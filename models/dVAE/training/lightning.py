import pytorch_lightning as pl
import torch.optim.optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from einops import rearrange
from models.dVAE.models.model import DiscreteVAE
from models.dVAE.training.config import DVAETrainingConfig
from models.dVAE.training.params_schedule import LinearScheduler, ExponentialScheduler


class DVAETrainModule(pl.LightningModule):
    """A PyTorch Lightning training module for the `DVAE`."""

    def __init__(self, *, dvae: DiscreteVAE | None = None, config: DVAETrainingConfig | None = None):
        """Init the DVAE training module.

        Args:
            dvae: An instance of the Discrete Variational Auto-Encoder to be used.
            config: The DVAE's training configuration. If `None`, default configuration
                will be used.
        """
        super().__init__()
        self.dvae = dvae
        self.step = 0
        self.config = DVAETrainingConfig() if config is None else config
        self._temperature_scheduler = ExponentialScheduler(
            start=self.config.temperature_scheduler.start,
            min=self.config.temperature_scheduler.min,
            anneal_rate=config.temperature_scheduler.anneal_rate
        )
        self._kl_div_weight_scheduler = LinearScheduler(
            start=self.config.kl_div_weight_scheduler.start,
            end=self.config.kl_div_weight_scheduler.end,
            warmup=self.config.kl_div_weight_scheduler.warmup,
            cooldown=self.config.kl_div_weight_scheduler.cooldown,
            steps=self.config.max_epochs,
        )
        self.save_hyperparameters(ignore=['dvae'])


    def configure_optimizers(self) -> tuple[list[AdamW], list[ExponentialLR]]:  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = ExponentialLR(
            optimizer,
            gamma= self.config.lr_decay_rate
        )
        return [optimizer], [scheduler]

    def _step(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        temperature = self._temperature_scheduler.get_value()
        kl_div_weight = self._kl_div_weight_scheduler.get_value()

        loss, recons = self.dvae(
            batch,
            return_loss = True,
            return_recons = True,
            temp = temperature
        )

        self._temperature_scheduler.step(self.step)

        self.log(f"{mode}/loss", loss, prog_bar=mode == "train")
        self.log("temperature", self._temperature_scheduler.get_value())
        self.log("kl_div_weight", self._kl_div_weight_scheduler.get_value())
        
        self.step += 1
        
        return loss
        
    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def on_train_epoch_end(self) -> None:  # noqa: D102
        self._kl_div_weight_scheduler.step()

    def validation_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, _: int) -> None:  # noqa: D102
        self._step(batch, "test")
