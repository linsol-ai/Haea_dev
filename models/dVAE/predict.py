import argparse
import logging

import pytorch_lightning as pl
import torch
from typing import List
import yaml
from pydantic import ValidationError
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
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

class VariableProprecess:
    def __init__(self, variables, model_path: str, year_offset: int ):
        input, target, normalizaion = self.load_dataset(variables, year_offset)


    def load_dataset(self, variables, year_offset):
        device = ("cuda" if torch.cuda.is_available() else "cpu" )
        self.device = torch.device(device)
        weather = WeatherDataset(year_offset, device=device, offline=True)
        return weather.load(variables=variables)

    def load_models(self, variables, model_path) -> List[DVAETrainModule]:
        for key in self.