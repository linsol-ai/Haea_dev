import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Dict
from torch.optim import Adam
import torch.nn.functional as F
from models.HAEA.models.model import Haea
from models.HAEA.training.configs import TrainingConfig
from models.HAEA.training.params_schedule import CosineWarmupScheduler
import wandb

def denormalize(inputs, mean_std) -> torch.Tensor:
    mean = mean_std[:, 0].view(1, 1, mean_std.size(0), 1)
    std = mean_std[:, 1].view(1, 1, mean_std.size(0), 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized

def rmse_loss(x, y):
    return torch.sqrt(F.mse_loss(x, y))


class TrainModule(pl.LightningModule):

    def __init__(self, *, model: Haea, mean_std: torch.Tensor,
                 max_iters: int, config: TrainingConfig):
        
        super().__init__()
        self.max_iters = max_iters
        self.model = model
        self.mean_std = mean_std
        self.config = config
        self.save_hyperparameters()

    
    def setup(self, stage: str) -> None:
        print(stage)
        self.mean_std = self.mean_std.to(self.device)


    def configure_optimizers(self) -> Adam:  # noqa: D102
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.config.warmup_step, max_iters=self.max_iters
        )
        return optimizer


    def _step(self, batch: Dict, mode: str) -> torch.Tensor:
        src = batch['source']
        src_id = batch['source_id']
        tgt = batch['target']
        tgt_id = batch['target_id']

        predict = self.model(src, src_id, tgt, tgt_id)

        loss = rmse_loss()

        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")
        return loss


    def calculate_rmse_loss(self, predict: torch.Tensor, label: torch.Tensor, var_len: int):
        # label.shape = (batch, time_len * var_len, hidden) -> not nomalized
        # predict.shape = (batch, time_len * var_len, hidden) -> not nomalized

        predict = predict.view(predict.size(0), -1, var_len, predict.size(2))
        # predict.shape = (batch, time_len, var_len, 1450) -> not nomalized
        reversed_predict = denormalize(predict, self.mean_std)

        label = label.view(label.size(0), -1, var_len, label.size(2))
        label = denormalize(label, self.mean_std)

        # reversed_predict.shape = (batch, time_len * var_len, 1450) -> nomalized
        loss = rmse_loss(reversed_predict, label)
        return loss


    def calculate_sqare_loss(self, predict: torch.Tensor, label: torch.Tensor):
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
        loss = F.mse_loss(reversed_predict, label, reduction='none')
        return loss
    

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
        

    def training_step(self, batch: Dict, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: Dict, _: int) -> None:  # noqa: D102
        self._step(batch, "test")

    

