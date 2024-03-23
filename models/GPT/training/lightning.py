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
        prd1, prd2 = self.model(src, self.var_list, self.tgt_mask)
        label1 = batch[:, 1:]
        label2 = batch[:, 2:]
        label1 = label1.view(label1.size(0), -1, label1.size(-1))
        label2 = label2.view(label2.size(0), -1, label2.size(-1))
        loss = rmse_loss(predict, label)
        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")

        return loss


    def calculate_rmse_loss(self, predict: torch.Tensor, label: torch.Tensor):
        # target.size = (batch, time_len, var_len, hidden)
        var_len = label.size(2)
        label = denormalize(label, self.mean_std)
        label = label.view(label.size(0), -1, label.size(3))

        # predict.shape = (batch, time_len * var_len, hidden) -> not nomalized
        predict = predict.view(predict.size(0), -1, var_len, predict.size(2))
        # predict.shape = (batch, time_len, var_len, 1450) -> not nomalized
        reversed_predict = denormalize(predict, self.mean_std)
        reversed_predict = reversed_predict.view(reversed_predict.size(0), -1, reversed_predict.size(3))
        # reversed_predict.shape = (batch, time_len * var_len, 1450) -> nomalized
        loss = rmse_loss(reversed_predict, label)
        return loss


    def calculate_sqare_loss(self, predict: torch.Tensor, label: torch.Tensor):
        # target.size = (batch, time_len, var_len, hidden)
        label = denormalize(label, self.mean_std)
        # predict.shape = (batch, time_len, var_len, 1450) -> not nomalized
        reversed_predict = denormalize(predict, self.mean_std)
        # reversed_predict.shape = (batch, time_len * var_len, 1450) -> nomalized
        loss = F.mse_loss(reversed_predict, label, reduction='none')
        reversed_predict.cpu().detach()
        predict.cpu().detach()
        reversed_predict.cpu().detach()
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
        self.tgt_mask = self.tgt_mask.to(self.device)
        self.model.eval()

    def forward(self, batch) -> torch.Tensor:
        src = batch[0].to(self.device)
        # (batch, time+1, var, hidden)
        label = batch[1].to(self.device)
        tgt = label[:, :-1]

        src_seq, tgt_seq = get_var_seq(self.src_var_list, self.tgt_var_list, self.config.src_time_len, self.config.tgt_time_len, src.size(0))
        src_seq = src_seq.to(self.device)
        tgt_seq = tgt_seq.to(self.device)
      
        # predict.shape = (batch, time * var + 1, hidden)
        predict = self.model(src, tgt, src_seq, tgt_seq, self.tgt_mask)
        label = label[:, 1:]
        predict = predict.view(predict.size(0), self.config.tgt_time_len, self.tgt_var_list.size(0), predict.size(-1))
        # loss.shape = (batch, time_len * var_len, 1450)
        loss = self.calculate_sqare_loss(predict, label)
        # loss.shape = (batch, var_len, time_len, 1450)
        loss = loss.swapaxes(1, 2)
        # loss.shape = (batch, var_len, time_len)
        loss = torch.mean(loss, dim=-1)
        # loss.shape = (var_len, batch, time_len)
        loss = loss.swapaxes(0, 1)

        src_seq.cpu().detach()
        tgt_seq.cpu().detach()
        src.cpu().detach()

        label.cpu().detach()
        predict.cpu().detach()
        loss = loss.cpu().detach()

        return loss
    

