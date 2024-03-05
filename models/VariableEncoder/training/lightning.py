import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
from torch.optim import Adam
import torch.nn.functional as F
from models.VariableEncoder.models.model import VariableEncoder
from models.VariableEncoder.training.configs import TrainingConfig
from models.VariableEncoder.training.params_schedule import CosineWarmupScheduler
import wandb
import math

def denormalize(inputs, mean_std) -> torch.Tensor:
    mean = mean_std[:, 0].view(1, 1, mean_std.size(0), 1)
    std = mean_std[:, 1].view(1, 1, mean_std.size(0), 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized

def rmse_loss(x, y):
    return torch.sqrt(F.mse_loss(x, y))


 def get_var_seq(self, batch_size):
        tgt_seq = self.tgt_var_list.repeat_interleave(self.tgt_time_len, dim=0).unsqueeze(0).repeat_interleave(batch_size, dim=0)
        src_seq = self.src_var_list.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        return src_seq, tgt_seq


def positional_encoding(self, batch, d_model, var_len, time_len):
        pe = torch.zeros(batch, time_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, time_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe.repeat_interleave(var_len, dim=1)
    

def get_tgt_mask(self) -> torch.Tensor:
        var_len = len(self.tgt_var_list)
        matrix = torch.zeros(var_len * self.tgt_time_len, var_len * self.tgt_time_len)

        for i in range(self.tgt_time_len):
            for _ in range(var_len):
                inf_idx = min(((i)*var_len), var_len * self.tgt_time_len)
                matrix[:(i*var_len), inf_idx:] = float('-inf')
        return matrix


class TrainModule(pl.LightningModule):

    def __init__(self, *, model: VariableEncoder, mean_std: torch.Tensor,
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


    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], mode: str) -> torch.Tensor:
        src = batch[0]
        tgt = batch[1]
        predict = self.model(src, tgt)
        loss = rmse_loss(predict, tgt.view(tgt.size(0), -1, tgt.size(3)))
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


    def visualization_air(self, air_loss: torch.Tensor):
        for i, name in enumerate(self.config.air_variable):
            start = i * self.pressure_level
            end = start + self.pressure_level
            loss = air_loss[start:end]
      

            custom_plot = wandb.plot.line_series(
                xs=range(loss.size(1)), 
                ys=loss,
                keys=range(self.pressure_level),
                title=name,
                xname="Time - 1Hour per"
            )

            self.logger.experiment.log({f"Atmospheric Loss/{name}": custom_plot})
    

    def visualization_surface(self, surface_loss: torch.Tensor):
        for i, name in enumerate(self.config.surface_variable):
            loss = surface_loss[i]
       

            custom_plot = wandb.plot.line_series(
                xs=range(loss.size(0)), 
                ys=[loss],
                title=name,
                xname="Time - 1Hour per"
            )

            self.logger.experiment.log({f"Surface Loss/{name}": custom_plot})
    

    def validation(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        src = batch[0].to(self.device)
        tgt = batch[1].to(self.device)

        var_len = tgt.size(2)
        predict = self.model(src, tgt)
        loss = self.calculate_sqare_loss(predict, tgt)

        # loss.shape = (batch, time_len, var_len, 1450)
        loss = loss.view(loss.size(0), -1, var_len, loss.size(2))
        # loss.shape = (batch, var_len, time_len, 1450)
        loss = loss.swapaxes(1, 2)
        hidden = loss.size(3)
        # loss.shape = (batch, var_len, time_len)
        loss = torch.sum(loss, dim=-1) / hidden

        # loss.shape = (var_len, batch, time_len)
        loss = loss.swapaxes(0, 1)
        n_batch = loss.size(1)
        # loss.shape = (var_len, time_len)
        loss = torch.sum(loss, dim=1) / n_batch
        loss = torch.sqrt(loss)

        air_loss = loss[:self.pressure_level * len(self.config.air_variable), :]
        surface_loss = loss[self.pressure_level * len(self.config.air_variable):, :]

        self.visualization_air(air_loss)
        self.visualization_surface(surface_loss)

        

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

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = batch[0].to(self.device)
        tgt = batch[1].to(self.device)
        var_len = tgt.size(2)
        predict = self.model(src, tgt)
        # loss.shape = (batch, time_len * var_len, 1450)
        loss = self.calculate_sqare_loss(predict, tgt)
        loss = loss.view(loss.size(0), -1, var_len, loss.size(2))
        # loss.shape = (batch, var_len, time_len, 1450)
        loss = loss.swapaxes(1, 2)
        # loss.shape = (batch, var_len, time_len)
        loss = torch.mean(loss, dim=-1)
        # loss.shape = (var_len, batch, time_len)
        loss = loss.swapaxes(0, 1)

        predict = predict.view(predict.size(0), -1, var_len, predict.size(2))
        predict = predict.swapaxes(1, 2)
        predict = torch.mean(predict, dim=-1)
        # predict.shape = (var_len, batch, time_len)
        predict = predict.swapaxes(0, 1)

        tgt = tgt.swapaxes(1, 2)
        tgt = torch.mean(tgt, dim=-1)
        tgt = tgt.swapaxes(0, 1)

        src.cpu().detach()
        tgt = tgt.cpu().detach()
        predict = predict.cpu().detach()
        loss = loss.cpu().detach()

        return loss, predict, tgt
    

