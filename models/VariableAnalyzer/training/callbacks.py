import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import wandb
from typing import List
from models.VariableAnalyzer.training.lightning import TrainModule

class VariableVaildationCallback(Callback):
    """Callback to save visualizations of a dataset throughout training.

    The callback will save
    - the first `n_images` images of the dataset auto-encoded by the model
    - the codebook vectors distribution.
    """

    def __init__(
        self, level_var: List, non_level_var: List, level_info: List, val_batch: int, log_every_n_step: int, dataset: Dataset, logger: WandbLogger
    ) -> None:
        """Init the callback.

        Args:
            n_images: The number of images to save.
            log_every_n_epochs: Log the visualization every `log_every_n_epochs` epochs.
            dataset: The dataset to visualize.
            logger: Th  e logger to be used to save the visualizations.
        """
        self.level_var = level_var
        self.non_level_var = non_level_var
        self.level_info = [f'{level} hPa' for level in level_info]
        self.val_batch = val_batch
        self.log_every_n_step = log_every_n_step
        self._dataset = dataset
        self._logger = logger


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 트레이닝 스텝이 100의 배수일 때 원하는 작업 수행
        if (trainer.global_step + 1) % self.log_every_n_step == 0:
            if pl_module.logger is None:
                raise ValueError("Logger is not set.")
            self._plot_predictions(pl_module, trainer.global_step + 1)
    

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

            self._logger.experiment.log({"my_custom_id": custom_plot})
    

    def 


        
    
    def validation(self, pl_module: TrainModule):
        src = torch.stack([self._dataset[i][0] for i in range(self.val_batch)], dim=0).to(
            pl_module.device  # type: ignore[arg-type]
        )
        tgt = torch.stack([self._dataset[i][1] for i in range(self.val_batch)], dim=0).to(
            pl_module.device  # type: ignore[arg-type]
        )
        label = torch.stack([self._dataset[i][2] for i in range(self.val_batch)], dim=0).to(
            pl_module.device  # type: ignore[arg-type]
        )
        predict = pl_module.model(src, tgt)
        loss = pl_module.calculate_loss(predict[:, :, :self.predict_dim], label[:, :, :self.predict_dim])

        # loss.shape = (batch, time_len, var_len, 1450)
        loss = loss.view(loss.size(0), -1, loss.size(3))
        # loss.shape = (batch, var_len, time_len, 1450)
        loss = loss.permute(0, 2, 1, 3)

        # loss.shape = (var_len, time_len)
        loss = torch.sum(loss, dim=0)
        loss = torch.sum(loss, dim=2)

        level_loss = loss[:, :13 * len(self.level_var)]
        non_level_loss = loss[:, 13 * len(self.level_var):]

        pl_module.log
        
       


        return loss