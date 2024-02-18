import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
from torch.optim import Adam
import torch.nn.functional as F
from models.VariableEncoder.models.model import VariableEncoder
from models.VariableEncoder.training.configs import TrainingConfig
from models.VariableEncoder.training.params_schedule import CosineWarmupScheduler
from typing import List
import wandb

def denormalize(inputs, mean_std):
    # min_max 텐서를 적절히 재구성하여 inputs의 차원에 맞춤
    mean = mean_std[:, 0].view(1, 1, mean_std.size(0), 1)
    std = mean_std[:, 1].view(1, 1, mean_std.size(0), 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized

def rmse_loss(x, y):
    return torch.sqrt(F.mse_loss(x, y))


class TrainModule(pl.LightningModule):

    def __init__(self, *, model: VariableEncoder, mean_std: torch.Tensor, var_len: int, 
                 max_iters: int, var_lv: List, var_nlv: List, levels: List, 
                 config: TrainingConfig | None = None):
        
        super().__init__()
        self.var_len = var_len
        self.var_lv = var_lv
        self.var_nlv = var_nlv
        self.levels = levels
        self.max_iters = max_iters
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
        loss = self.calculate_rmse_loss(predict, label)
        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")
        return loss
    
    def on_save_checkpoint(self, checkpoint):
        # dvae 상태를 checkpoint 딕셔너리에 추가
        checkpoint['model_state'] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # checkpoint 딕셔너리에서 dvae 상태를 로드
        self.model.load_state_dict(checkpoint['model_state'])


    def calculate_rmse_loss(self, predict: torch.Tensor, label: torch.Tensor):
        has_nan = torch.isnan(predict).any()
        if has_nan:
            
        # predict.shape = (batch, time_len * var_len, 1450) -> not nomalized
        predict = predict.view(predict.size(0), -1, self.var_len, predict.size(2))
        # predict.shape = (batch, time_len, var_len, 1450) -> not nomalized
        reversed_predict = denormalize(predict, self.mean_std)
        reversed_predict = reversed_predict.view(reversed_predict.size(0), -1, reversed_predict.size(3))
        # reversed_predict.shape = (batch, time_len * var_len, 1450) -> nomalized
        loss = rmse_loss(reversed_predict, label)
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration


    def visualization_level(self, level_loss: torch.Tensor):
        for i in range(len(self.var_lv)):
            start = i * len(self.levels)
            end = start + len(self.levels)
            name = self.var_lv[i]
            loss = level_loss[start:end]

            custom_plot = wandb.plot.line_series(
                xs=range(loss.size(1)), 
                ys=loss,
                keys=self.levels,
                title=name,
                split_table=True,
                xname="Time - 6Hour per"
            )

            self.logger.experiment.log({f"Atmospheric Loss/{name}": custom_plot})
    

    def visualization_non_level(self, non_level_loss):
        for i in range(len(self.var_nlv)):
            name = self.var_nlv[i]
            loss = non_level_loss[i]

            custom_plot = wandb.plot.line_series(
                xs=range(loss.size(0)), 
                ys=[loss],
                title=name,
                xname="Time - 6Hour per"
            )

            self.logger.experiment.log({f"Surface Loss/{name}": custom_plot})
    

    def validation(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        src = batch[0]
        tgt = batch[1]
        label = batch[2]
        label = label.view(label.size(0), -1, label.size(3))
        predict = self.model(src, tgt)
        loss = self.calculate_sqare_loss(predict, label)
        # loss.shape = (batch, time_len, var_len, 1450)
        loss = loss.view(loss.size(0), -1, self.var_len, loss.size(2))
        # loss.shape = (batch, var_len, time_len, 1450)
        loss = loss.permute(0, 2, 1, 3)
        hidden = loss.size(3)
        # loss.shape = (batch, var_len, time_len)
        loss = torch.sum(loss, dim=-1) / hidden
        # loss.shape = (var_len, batch, time_len)
        loss = loss.permute(1, 0, 2)
        n_batch = loss.size(1)
        # loss.shape = (var_len, time_len)
        loss = torch.sum(loss, dim=1) / n_batch
        loss = torch.sqrt(loss)

        print(loss.shape)

        level_loss = loss[:len(self.levels) * len(self.var_lv), :]
        non_level_loss = loss[len(self.levels) * len(self.var_lv):, :]

        self.visualization_level(level_loss)
        self.visualization_non_level(non_level_loss)


    def calculate_sqare_loss(self, predict: torch.Tensor, label: torch.Tensor):
        # predict.shape = (batch, time_len * var_len, 1450) -> not nomalized
        predict = predict.view(predict.size(0), -1, self.var_len, predict.size(2))
        # predict.shape = (batch, time_len, var_len, 1450) -> not nomalized
        reversed_predict = denormalize(predict, self.mean_std)
        reversed_predict = reversed_predict.view(reversed_predict.size(0), -1, reversed_predict.size(3))
        # reversed_predict.shape = (batch, time_len * var_len, 1450) -> nomalized
        loss = F.mse_loss(reversed_predict, label, reduction='none')
        return loss
        

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: D102
        if batch_idx == 0:
            self.validation(batch)
            return
        return self._step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _: int) -> None:  # noqa: D102
        self._step(batch, "test")
    

