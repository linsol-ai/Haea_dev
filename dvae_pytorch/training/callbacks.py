import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import wandb

from training.lightning import DVAETrainModule


class SaveValVisualizationCallback(Callback):
    """Callback to save visualizations of a dataset throughout training.

    The callback will save
    - the first `n_images` images of the dataset auto-encoded by the model
    - the codebook vectors distribution.
    """

    def __init__(
        self, n_images: int, log_every_n_step: int, dataset: Dataset, logger: WandbLogger
    ) -> None:
        """Init the callback.

        Args:
            n_images: The number of images to save.
            log_every_n_epochs: Log the visualization every `log_every_n_epochs` epochs.
            dataset: The dataset to visualize.
            logger: The logger to be used to save the visualizations.
        """
        self._n_images = n_images
        self.log_every_n_step = log_every_n_step
        self._dataset = dataset
        self._logger = logger


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 트레이닝 스텝이 100의 배수일 때 원하는 작업 수행
        if (trainer.global_step + 1) % self.log_every_n_step == 0:
            if pl_module.logger is None:
                raise ValueError("Logger is not set.")
            self._plot_predictions(pl_module, trainer.global_step + 1)
        
    
    def _plot_histogram_codes(self, code, step):
        hist_plot = wandb.Histogram(code)
        self._logger.experiment.log({"val/codebook_indices": hist_plot}, step)


    def _plot_predictions(self, pl_module: DVAETrainModule, step) -> None:
        img = torch.stack([self._dataset[i] for i in range(self._n_images)], dim=0).to(
            pl_module.device  # type: ignore[arg-type]
        )
        with torch.no_grad():
            temperature = pl_module._temperature_scheduler.get_value()
            code, logits = pl_module.dvae.get_codebook_indices(img)
            hard_recons = pl_module.dvae.decode(code, logits.shape[2:]).detach().cpu()
            recons = pl_module.dvae(
                img,
                return_loss = False,
                return_recons = True,
                temp = temperature
            ).detach().cpu()
            img = img.detach().cpu()
            code = code.detach().cpu()

            grid = make_grid(torch.concat([img, recons, hard_recons], dim=0), nrow=self._n_images)
            self._logger.log_image("val/predictions", [grid], step)
            self._plot_histogram_codes(code, step)