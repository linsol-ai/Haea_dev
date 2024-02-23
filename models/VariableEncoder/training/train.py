import logging
import pytorch_lightning as pl
import torch
import yaml
from typing import List
from pydantic import ValidationError
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from absl import app
from pytorch_lightning.utilities.model_summary import ModelSummary

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
from datasets.weather_bench import WeatherDataset
from models.VariableEncoder.datasets.dataset import CustomDataset
from models.VariableEncoder.models.model import VariableEncoder

from models.VariableEncoder.training.configs import TrainingConfig
from models.VariableEncoder.training.configs import TrainingRunConfig
from models.VariableEncoder.training.lightning import TrainModule


def get_normal_dataset(config: TrainingConfig, year_offset: int, tgt_time_len: int):
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(year_offset, device=device)
    # dataset.shape:  torch.Size([7309, 100, 1450])
    source, label, mean_std = weather.load(config.air_variable, config.surface_variable, config.only_input_variable, config.constant_variable)
    dataset = CustomDataset(source, label, tgt_time_len, n_only_input=len(config.only_input_variable)+len(config.constant_variable))
    src_var_list = weather.get_var_code(config.air_variable, config.surface_variable + config.only_input_variable+config.constant_variable)
    tgt_var_list = weather.get_var_code(config.air_variable, config.surface_variable)
    return dataset, mean_std, (src_var_list, tgt_var_list)

    
if __name__ == "__main__":
    app.run(_main)
