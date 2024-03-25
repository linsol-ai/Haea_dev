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
        self.automatic_optimization = False
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

    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        

    def training_step(self, batch: torch.Tensor, _: int):
        optimizer = self.optimizers()
        src = batch[:, :-2]
        # predict.shape = (batch, time * var, hidden)
        pred = self.model(src, self.var_list, self.tgt_mask)
        label = batch[:, 1:-1]
        pred = pred.view(pred.size(0), self.config.time_len, self.var_list.size(0), pred.size(2))
        loss1 = F.mse_loss(pred, label)
        loss1.requires_grad_(True)

        optimizer.zero_grad()
        self.manual_backward(loss1)
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer.step()
        self.lr_scheduler.step()

        pred = self.model(pred.detach(), self.var_list, self.tgt_mask)
        label = batch[:, 2:]
        pred = pred.view(pred.size(0), self.config.time_len, self.var_list.size(0), pred.size(2))
        loss2 = F.mse_loss(pred, label)
        loss2.requires_grad_(True)

        optimizer.zero_grad()
        self.manual_backward(loss2)
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer.step()
        self.lr_scheduler.step()

        self.log(f"train/mse_loss1", loss1, prog_bar=True)
        self.log(f"train/mse_loss2", loss2, prog_bar=True)


    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        with torch.no_grad():
            src = batch[:, :-2]
            # predict.shape = (batch, time * var, hidden)
            pred = self.model(src, self.var_list, self.tgt_mask)
            label = batch[:, 1:-1]
            pred = pred.view(pred.size(0), self.config.time_len, self.var_list.size(0), pred.size(2))
            loss1 = F.mse_loss(pred, label)

            pred = self.model(pred, self.var_list, self.tgt_mask)
            label = batch[:, 2:]
            pred = pred.view(pred.size(0), self.config.time_len, self.var_list.size(0), pred.size(2))
            loss2 = F.mse_loss(pred, label)

            self.log(f"val/mse_loss1", loss1, prog_bar=False)
            self.log(f"val/mse_loss2", loss2, prog_bar=False)


    def test_step(self, batch: torch.Tensor, _: int) -> None:  # noqa: D102
        with torch.no_grad():
            src = batch[:, :-2]
            # predict.shape = (batch, time * var, hidden)
            pred = self.model(src, self.var_list, self.tgt_mask)
            label = batch[:, 1:-1]
            pred = pred.view(pred.size(0), self.config.time_len, self.var_list.size(0), pred.size(2))
            loss1 = F.mse_loss(pred, label)

            pred = self.model(pred, self.var_list, self.tgt_mask)
            label = batch[:, 2:]
            pred = pred.view(pred.size(0), self.config.time_len, self.var_list.size(0), pred.size(2))
            loss2 = F.mse_loss(pred, label)

            self.log(f"test/mse_loss1", loss1, prog_bar=False)
            self.log(f"test/mse_loss2", loss2, prog_bar=False)



    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def setting(self):
        self.mean_std = self.mean_std.to(self.device)
        self.tgt_mask = self.tgt_mask.to(self.device)
        self.var_list = self.var_list.to(self.device)
        self.model.eval()

    def forward(self, batch: torch.Tensor, max_lead_time: int, location=None) -> torch.Tensor:
        with torch.no_grad():
            src = batch[:, :-1]
            label = batch[:, 1:]
            # predict.shape = (batch, time * var, hidden)
            pred = self.model(src, self.var_list, self.tgt_mask)

            label = denormalize(label, self.mean_std)
            predict_all = denormalize(predict_all, self.mean_std)

            loss = F.mse_loss(predict_all, label, reduction='none')
            # loss.shape = (batch, max_lead_time, var_len, hidden)

            if location is not None:
                loss = loss[:, :, :, location]
                if len(loss.shape) == 4:
                    loss = loss.mean(dim=-1)
            else:
                loss = loss.mean(dim=-1)

            return loss
    

