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

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from datasets.weather_bench import WeatherDataset
from models.VariableAnalyzer.datasets.dataset import CustomDataset
from models.VariableAnalyzer.models.model import VariableAnalyzer
from models.VariableAnalyzer.training.configs import TrainingRunConfig
from models.VariableAnalyzer.training.lightning import TrainModule


def get_dataset(year_offset: int):
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(year_offset, device=device)
    # dataset.shape:  torch.Size([7309, 100, 1450])
    original = weather.load()
    dataset = CustomDataset

        

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
        model = VariableAnalyzer(
            
        )
        model_pl = DVAETrainModule(dvae=model, config=config.training)
        summary = ModelSummary(model_pl, max_depth=-1)
        print(summary)

        levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        variable_keys = ['geopotential', 'specific_humidity', 'temperature', 'vertical_velocity']

        weather = WeatherDataset(url='gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr')
        weather.load_init()

        start_date = pd.to_datetime('2022-01-01')
        end_date = pd.to_datetime('2022-09-01')

        output = weather.load_unet(variable_keys[0], levels, start_date, end_date, normalize=True)

        # Use a custom dataset class with proper transformations

        dataset = ImageDataset(output)
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
