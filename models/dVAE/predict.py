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
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from datasets.weather_bench import WeatherDataset
from models.dVAE.training.callbacks import SaveValVisualizationCallback
from models.dVAE.training.config import TrainingRunConfig
from models.dVAE.training.lightning import DVAETrainModule
from models.dVAE.models.model import DiscreteVAE

class VariableProprecess:
    
    def __init__(self, ):
