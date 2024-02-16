import argparse
import logging
from pathlib import Path
import pandas as pd

import pytorch_lightning as pl
import torch
import yaml
from pydantic import ValidationError
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.utilities.model_summary import ModelSummary

from training.callbacks import SaveValVisualizationCallback
from training.config import TrainingRunConfig
from training.lightning import DVAETrainModule
from models.model import DiscreteVAE

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from datasets.weather_bench import WeatherDataset

class ImageDataset(Dataset):
        def __init__(self, data_array: torch.Tensor):
            if len9data_array.shape
            self.data_array = data_array

        def __len__(self):
            return len(self.data_array)

        def __getitem__(self, idx):
            sample = self.data_array[idx]
            return sample
        

def _main() -> None:
    config_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'configs/dvae_weather_config.yaml')
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config: TrainingRunConfig = TrainingRunConfig.parse_obj(config_dict)
    except FileNotFoundError:
        logging.error(f"Config file {config_path} does not exist. Exiting.")
    except yaml.YAMLError:
        logging.error(f"Config file {config_path} is not valid YAML. Exiting.")
    except ValidationError as e:
        logging.error(f"Config file {config_path} is not valid. Exiting.\n{e}")
    else:
        pl.seed_everything(config.seed)
        logger = WandbLogger(save_dir=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'tb_logs'), name="my_model")
        model = DiscreteVAE(
            num_tokens=config.model.codebook_size,
            codebook_dim=config.model.codebook_vector_dim,
            num_layers=config.model.num_layers,
            num_resnet_blocks=config.model.num_resnet_blocks,
            hidden_dim=config.model.hidden_dim,
            channels=1,
            smooth_l1_loss=True,
            temperature=config.training.temperature_scheduler.start,
            straight_through=False
        )
        model_pl = DVAETrainModule(dvae=model, config=config.training)
        summary = ModelSummary(model_pl, max_depth=-1)
        print(summary)

        device = ("cuda" if torch.cuda.is_available() else "cpu" )
        device = torch.device(device)
        weather = WeatherDataset(0, device=device, offline=True)
        input, _, _ = weather.load()

        val_dataset = input[config.training.train_variable]

        # Use a custom dataset class with proper transformations

        dataset = ImageDataset(val_dataset)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset,
            [0.7, 0.3],
        )
        test_ds, val_ds = torch.utils.data.random_split(
            val_ds,
            [0.5, 0.5],
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.training.batch_size, num_workers=8, shuffle=True
        )
        test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, num_workers=4)

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=config.training.max_epochs,
            logger=logger,
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                SaveValVisualizationCallback(
                    n_images=config.training.num_vis,
                    log_every_n_step=config.training.save_vis_every_n_step,
                    dataset=train_ds,
                    logger=logger,
                ),
            ],
            precision="bf16"
        )

        trainer.fit(model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model_pl, dataloaders=test_loader)


if __name__ == "__main__":
    _main()
