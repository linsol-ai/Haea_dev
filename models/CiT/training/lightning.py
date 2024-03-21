import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
from torch.optim import Adam
import torch.nn.functional as F
from models.CiT.models.model import ClimateTransformer
from models.CiT.training.configs import TrainingConfig
from models.CiT.training.params_schedule import CosineWarmupScheduler
import wandb

def denormalize(inputs, mean_std) -> torch.Tensor:
    mean = mean_std[:, 0].view(1, mean_std.size(0), 1)
    std = mean_std[:, 1].view(1, mean_std.size(0), 1)
    # 역정규화 수행
    denormalized = (inputs * std) + mean
    return denormalized


def sigmoid_function(x):
    x = x / 300
    return 1 - torch.abs(1/(1+torch.exp(-x)) - 0.5)


def mse_loss(x, y, delta):
    # MSE 손실을 계산합니다. reduction='none'은 각 요소의 손실을 유지합니다.
    mse = F.mse_loss(x, y, reduction='none')
    # 배치 차원을 제외한 나머지 차원에 대해 평균을 계산합니다.
    # 여기서는 mse의 모든 차원에 대해 mean을 사용하지만, 배치 차원이 아닌 차원에만 적용됩니다.
    mean_mse = mse.mean(dim=list(range(1, mse.ndim)))
    weight = sigmoid_function(delta)
    # 최종 RMSE를 계산합니다.
    return mean_mse * weight


class TrainModule(pl.LightningModule):

    def __init__(self, *, model: ClimateTransformer, mean_std: torch.Tensor, var_list: torch.Tensor,
                 max_iters: int, config: TrainingConfig):
        
        super().__init__()
        self.max_iters = max_iters
        self.model = model
        self.var_list = var_list
        self.mean_std = mean_std
        self.config = config
        self.save_hyperparameters()

    
    def setup(self, stage: str) -> None:
        print(stage)
        self.mean_std = self.mean_std.to(self.device)
        self.var_list = self.var_list.to(self.device)


    def configure_optimizers(self) -> Adam:  # noqa: D102
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.config.warmup_step, max_iters=self.max_iters
        )
        return optimizer


    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], mode: str) -> torch.Tensor:
        src = batch[0]
        label = batch[1]
        delta = batch[2]

        label = label.view(label.size(0), -1, label.size(-1))
        predict = self.model(src, delta, self.var_list)
        loss = F.mse_loss(predict, label)

        self.log(f"{mode}/mse_loss", loss, prog_bar=mode == "train")
        return loss


    def calculate_sqare_loss(self, predict: torch.Tensor, label: torch.Tensor):
        # target.size = (batch, var_len, hidden)
        label = denormalize(label, self.mean_std)
        # reversed_predict.shape = (batch, var_len, hidden) -> nomalized
        reversed_predict = denormalize(predict, self.mean_std)
        loss = F.mse_loss(reversed_predict, label, reduction='none')
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

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], location=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            src = batch[0].to(self.device)
            src = src.squeeze(0)

            delta = batch[2].to(self.device)
            delta = delta.squeeze(0)

            var_seq = batch[3].to(self.device)

            predict = self.model(src, delta, var_seq)
            predict = predict.view(predict.size(0), self.config.time_len, -1, predict.size(-1))

            label = batch[1].to(self.device)
            label = label.squeeze(0)

            label = denormalize(label, self.mean_std)
            predict = denormalize(predict, self.mean_std)

            idx = (len(self.config.air_variable) * len(self.config.levels)) + self.config.surface_variable.index('total_precipitation')
            
            if location is not None:
                p_pred = predict[:, :, idx, location] * 1000
                p_label = label[:, :, idx, location] * 1000
                if len(p_pred.shape) == 3:
                    p_pred = p_pred.mean(dim=-1)
                    p_label = p_label.mean(dim=-1)
            else:
                p_pred = predict[:, :, idx] * 1000
                p_label = label[:, :, idx] * 1000
                p_pred = p_pred.mean(dim=-1)
                p_label = p_label.mean(dim=-1)

            # loss.shape = (batch, time_len)

            # loss.shape = (batch, time_len, var_len, hidden)
            loss = F.mse_loss(predict, label, reduction='none')
            # loss.shape = (batch(lead_days), time_len, var_len, hidden)

            if location is not None:
                loss = loss[:, :, :, location]
                if len(loss.shape) == 4:
                    loss = loss.mean(dim=-1)
            else:
                loss = loss.mean(dim=-1)
        
        return loss, p_pred, p_label
    

