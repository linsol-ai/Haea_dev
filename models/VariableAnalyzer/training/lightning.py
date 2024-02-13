import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from models.VariableAnalyzer.models.model import VariableAnalyzer
from models.VariableAnalyzer.training.configs import TrainingConfig
from models.VariableAnalyzer.training.params_schedule import CosineWarmupScheduler


def denormalize(inputs, mean_std):
    # min_max 텐서를 적절히 재구성하여 inputs의 차원에 맞춤
    mean = mean_std[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, var_len, 1)
    std = mean_std[:, 1].unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, var_len, 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized

def rmse_loss(x, y):
    return torch.sqrt(F.mse_loss(x, y))


class TrainModule(pl.LightningModule):

    def __init__(self, *, model: VariableAnalyzer, mean_std: torch.Tensor, var_len: int, 
                 predict_dim: int, max_iters: int, var_lvconfig: TrainingConfig | None = None):
        
        super().__init__()
        self.var_len = var_len
        self.max_iters = max_iters
        self.predict_dim = predict_dim
        self.model = model
        self.mean_std = mean_std
        self.config = TrainingConfig() if config is None else config
        self.save_hyperparameters(self.config.dict(), ignore=["model", "config"])

    
    def setup(self, stage: str) -> None:
        print(stage)
        self.model.init_seq(self.device)
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
        label = batch[2]
        label = label.view(label.size(0), -1, label.size(3))
        predict = self.model(src, tgt)
        loss = self.calculate_rmse_loss(predict[:, :, :self.predict_dim], label[:, :, :self.predict_dim])
        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")
        return loss


    def calculate_rmse_loss(self, predict: torch.Tensor, label: torch.Tensor):
        # predict.shape = (batch, time_len * var_len, 1450) -> not nomalized
        predict = predict.view(predict.size(0), -1, self.var_len, predict.size(2))
        # predict.shape = (batch, time_len, var_len, 1450) -> not nomalized
        mean_std = self.mean_std[0]
        reversed_predict = denormalize(predict, mean_std)
        reversed_predict = reversed_predict.view(reversed_predict.size(0), -1, reversed_predict.size(3))
        # reversed_predict.shape = (batch, time_len * var_len, 1450) -> nomalized
        loss = rmse_loss(reversed_predict, label)
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


    def visualization_level(self, level_loss):
        for i in range(len(self.level_var)):
            start = i * 13
            end = start + 13
            name = self.level_var[i]
            loss = level_loss[start:end]

            custom_plot = wandb.plot.line_series(
                xs=range(loss.size(1)), 
                ys=loss,
                keys=self.level_info,
                title=name,
                xname="Time - 6Hour per"
            )

            self._logger.experiment.log({f"Atmospheric Loss/{name}": custom_plot})
    

    def visualization_non_level(self, non_level_loss):
        for i in range(len(self.non_level_var)):
            name = self.non_level_var[i]
            loss = non_level_loss[i]

            data = [[x, y] for (x, y) in zip(range(loss.size(0)), loss)]
            table = wandb.Table(data=data, columns = ["avg loss", "Time - 6Hour per"])

            custom_plot = wandb.plot.line(
                table,
                "avg loss", "Time - 6Hour per",
                stroke=None,
                title=name,
            )

            self._logger.experiment.log({f"Surface Loss/{name}": custom_plot})
    

    def validation(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        src = batch[0]
        tgt = batch[1]
        label = batch[2]
        label = label.view(label.size(0), -1, label.size(3))
        predict = self.model(src, tgt)
        loss = self.calculate_rmse_loss(predict[:, :, :self.predict_dim], label[:, :, :self.predict_dim])
        # loss.shape = (batch, time_len, var_len, 1450)
        loss = loss.view(loss.size(0), -1, self.var_len, loss.size(2))
        # loss.shape = (batch, var_len, time_len, 1450)
        loss = loss.permute(0, 2, 1, 3)

        
    

    def calculate_mse_loss(self, predict: torch.Tensor, label: torch.Tensor):
        # predict.shape = (batch, time_len * var_len, 1450) -> not nomalized
        predict = predict.view(predict.size(0), -1, self.var_len, predict.size(2))
        # predict.shape = (batch, time_len, var_len, 1450) -> not nomalized
        mean_std = self.mean_std[0]
        reversed_predict = denormalize(predict, mean_std)
        reversed_predict = reversed_predict.view(reversed_predict.size(0), -1, reversed_predict.size(3))
        # reversed_predict.shape = (batch, time_len * var_len, 1450) -> nomalized
        loss = F.mse_loss(reversed_predict, label, reduction='none')
        return loss



        

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> None:  # noqa: D102
        self._step(batch, "test")
