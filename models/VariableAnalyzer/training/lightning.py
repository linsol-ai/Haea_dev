import pytorch_lightning as pl
import torch.optim.optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torch.optim.ex
from einops import rearrange
import pytorch_lightning
from models.dalle import DiscreteVAE
from training.config import DVAETrainingConfig
from training.params_schedule import LinearScheduler, ExponentialScheduler


class DVAETrainModule(pl.LightningModule):
    """A PyTorch Lightning training module for the `DVAE`."""

    def __init__(self, *, dvae: DiscreteVAE, config: DVAETrainingConfig | None = None):
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
        self.save_hyperparameters(self.config.dict(), ignore=["dvae", "config"])


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)  # 초기 learning rate 설정

        # Warmup을 위한 scheduler 정의
        warmup_steps = 4000
        warmup_scheduler = LambdaLR(optimizer, lambda step: min(step/warmup_steps, 1))

        # Cosine decay를 위한 scheduler 정의
        total_steps = 10000  # 전체 학습 step 예상치
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

        # PyTorch Lightning은 scheduler를 'scheduler' 키와 함께 반환할 때, 'interval'과 'frequency' 옵션을 제공합니다.
        # 'interval'은 'step' 또는 'epoch'일 수 있으며, 'frequency'는 해당 작업을 수행할 빈도입니다.
        # 여기서는 'step'마다 scheduler를 업데이트합니다.
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': warmup_scheduler, 'interval': 'step', 'frequency': 1},
                'lr_scheduler': {'scheduler': decay_scheduler, 'interval': 'step', 'frequency': 1, 'start_step': warmup_steps}}

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
