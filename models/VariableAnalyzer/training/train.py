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
from absl import app
from absl import flags
from pytorch_lightning.utilities.model_summary import ModelSummary

import sys,os
current_directory = os.getcwd()
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
from datasets.weather_bench import WeatherDataset
from models.VariableAnalyzer.datasets.dataset import CustomDataset
from models.VariableAnalyzer.models.model import VariableAnalyzer
from models.VariableAnalyzer.training.configs import TrainingRunConfig
from models.VariableAnalyzer.training.lightning import TrainModule

FLAGS = flags.FLAGS
YEAR_OFFSET = flags.DEFINE_integer('train_offset', None, help='training year')
TIME_LENGTH = flags.DEFINE_integer('time_len', None, help='TIME_LENGTH')
flags.mark_flag_as_required("train_offset")
flags.mark_flag_as_required("time_len")


def get_dataset(year_offset: int, time_len: int):
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(year_offset, device=device)
    # dataset.shape:  torch.Size([7309, 100, 1450])
    original, dims = weather.load()
    dataset = CustomDataset(original, time_len, device=device)
    return dataset, original.shape, dims

        

def _main(args) -> None:
    config_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'configs/train_config.yaml')
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

        train_offset = FLAGS.train_offset
        time_len = 4 * FLAGS.time_len

        # shape = (time, var, hidden)
        dataset, shape, dims = get_dataset(train_offset, time_len)

        logger = WandbLogger(save_dir=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'tb_logs'), name="my_model")
        model = VariableAnalyzer(
            var_len=shape[1],
            time_len=time_len,
            dim_model=shape[2],
            predict_dim=
            num_heads=config.model.num_heads,
            n_encoder_layers=config.model.n_encoder_layers,
            n_decoder_layers=config.model.n_decoder_layers,
            dropout=config.model.dropout
        )
        

        # Use a custom dataset class with proper transformations

        train_ds, test_ds = torch.utils.data.random_split(
            dataset,
            [0.7, 0.3],
        )
        val_ds, test_ds = torch.utils.data.random_split(
            test_ds,
            [0.3, 0.7],
        )   

        train_loader = DataLoader(
            train_ds, batch_size=config.training.batch_size, shuffle=True, pin_memory=True
        )
        test_loader = DataLoader(test_ds, batch_size=config.training.batch_size)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size)

        model_pl = TrainModule(model=model, max_iters=config.training.max_epochs*len(train_loader), config=config.training)
        summary = ModelSummary(model_pl, max_depth=-1)
        print(summary)

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=config.training.max_epochs,
            logger=logger,
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
            ],
            precision="bf16"
        )

        trainer.fit(model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model_pl, dataloaders=test_loader)


if __name__ == "__main__":
    app.run(_main)
