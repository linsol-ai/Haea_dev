import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
from torch.optim import Adam
import torch.nn.functional as F
from models.GPT.models.models import CliGPT
from models.GPT.training.configs import TrainingConfig
from models.GPT.training.params_schedule import CosineWarmupScheduler
import wandb
import math


def denormalize(inputs, mean_std) -> torch.Tensor:
    mean = mean_std[:, 0].view(1, 1, mean_std.size(0), 1)
    std = mean_std[:, 1].view(1, 1, mean_std.size(0), 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized

def get_tgt_mask(var_len, time_len) -> torch.Tensor:
    size = var_len * time_len
    matrix = torch.full((size, size), float('-inf'), dtype=torch.get_default_dtype())
    for i in range(time_len):
        s =  (i * var_len)
        e =  ((i+1) * var_len)
        matrix[s:e, :e] = 0
    return matrix


class TrainModule(pl.LightningModule):

    def __init__(self, *, model: CliGPT, mean_std: torch.Tensor, var_list: torch.Tensor,
                 max_iters: int, config: TrainingConfig):
        
        super().__init__()
        self.max_iters = max_iters
        self.model = model
        self.mean_std = mean_std
        self.config = config
        self.var_list = var_list
        self.save_hyperparameters()
        self.tgt_mask = get_tgt_mask(var_list.size(0), config.time_len)


    def setup(self, stage: str) -> None:
        self.mean_std = self.mean_std.to(self.device)
        self.tgt_mask = self.tgt_mask.to(self.device)
        self.var_list = self.var_list.to(self.device)


    def configure_optimizers(self) -> Adam:  # noqa: D102
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.config.warmup_step, max_iters=self.max_iters
        )
        return optimizer


    def _step(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        src = batch[:, :-2]
        # predict.shape = (batch, time * var + 1, hidden)
        pred = self.model(src, self.var_list, self.tgt_mask)
        label = batch[:, 2:]
        label = label.view(label.size(0), -1, label.size(-1))
        loss = F.mse_loss(pred, label)
        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")

        return loss

    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
        

    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, _: int) -> None:  # noqa: D102
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def setting(self):
        self.mean_std = self.mean_std.to(self.device)
        self.tgt_mask = self.tgt_mask.to(self.device)
        self.model.eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch = batch.to(self.device)
            src = batch[:self.config.time_len]
            for i in range(src.size(1)-self.config.time_len):
                predict = self.model(src, self.var_list, self.tgt_mask)
                token = predict[:, :-self.var_list.size(0)]
                
            

            return loss
    

