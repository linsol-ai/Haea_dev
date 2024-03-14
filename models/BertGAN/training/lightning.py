import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
from torch.optim import Adam
import torch.nn.functional as F
from models.BertGAN.model.model import Discriminator, Generator
from models.BertGAN.training.configs import TrainingConfig
from models.BertGAN.training.params_schedule import CosineWarmupScheduler
import wandb

def denormalize(inputs, mean_std) -> torch.Tensor:
    mean = mean_std[:, 0].view(1, mean_std.size(0), 1)
    std = mean_std[:, 1].view(1, mean_std.size(0), 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized

def rmse_loss(x, y):
    return torch.sqrt(F.mse_loss(x, y))


class BertGAN(pl.LightningModule):

    def __init__(self, *, generator: Generator, discriminator: Discriminator, var_list: torch.Tensor,
                 mean_std: torch.Tensor, max_iters: int, config: TrainingConfig):
        
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.max_iters = max_iters
        self.mean_std = mean_std
        self.config = config
        self.automatic_optimization = False
        self.save_hyperparameters()

    
    def setup(self, stage: str) -> None:
        print(stage)
        self.mean_std = self.mean_std.to(self.device)


    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)

        self.g_scheduler = CosineWarmupScheduler(
            g_opt, warmup=self.config.warmup_step, max_iters=self.max_iters
        )

        self.d_scheduler = CosineWarmupScheduler(
            d_opt, warmup=self.config.warmup_step, max_iters=self.max_iters
        )
        
        return g_opt, d_opt


    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], mode: str) -> torch.Tensor:
        src = batch[0]
        label = batch[1]
        delta = batch[2]
        var_seq = batch[3]
        label = label.view(label.size(0), -1, label.size(-1))
        predict = self.model(src, delta, var_seq)
        loss = rmse_loss(predict, label)
        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")
        return loss

    

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> None:  # noqa: D102
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def setting(self):
        self.mean_std = self.mean_std.to(self.device)
        self.model.eval()

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            src = batch[0].to(self.device)
            delta = batch[2].to(self.device)
            var_seq = batch[3].to(self.device)
            predict = self.model(src, delta, var_seq)
            predict = predict.view(predict.size(0), self.config.time_len, -1, predict.size(-1))

            label = batch[1].to(self.device)
            label = denormalize(label, self.mean_std)
            predict = denormalize(predict, self.mean_std)
            # loss.shape = (batch, time_len, var_len, hidden)
            loss = F.mse_loss(predict, label, reduction='none')
            # loss.shape = (batch, time_len, var_len)
            loss = loss.mean(dim=-1)
        
        return loss, delta
    

