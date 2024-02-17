import argparse
import logging

import pytorch_lightning as pl
import torch
import yaml
from pydantic import ValidationError
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
from datasets.weather_bench import WeatherDataset

from models.dVAE.training.callbacks import SaveValVisualizationCallback
from models.dVAE.training.config import TrainingRunConfig
from models.dVAE.training.lightning import DVAETrainModule
from models.dVAE.models.model import DiscreteVAE

class ImageDataset(Dataset):
        def __init__(self, data_array: torch.Tensor):
            if len(data_array.shape) == 5:
                self.data_array = data_array.view(-1, 1, data_array.size(3), data_array.size(4))
            else:
                self.data_array = data_array

        def __len__(self):
            return len(self.data_array)

        def __getitem__(self, idx):
            sample = self.data_array[idx]
            return sample
        

def train(var_key:str) -> None:

        

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

        device = ("cuda" if torch.cuda.is_available() else "cpu" )
        device = torch.device(device)
        weather = WeatherDataset(0, device=device, offline=True)

        vars = weather.HAS_LEVEL_VARIABLE + weather.NONE_LEVEL_VARIABLE
        input, _, _ = weather.load(variables=vars)



        


if __name__ == "__main__":
    _main()
