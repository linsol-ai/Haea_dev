import argparse
import logging
import tqdm
import pytorch_lightning as pl
import torch
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
from training.lightning import TrainModule

class VariablePredictor:
    def __init__(self, model_path: str, batch_size: int):
        self.model_path = model_path
        self.batch_size
    
    def load_models(self, model_path):
        folder_path = Path(model_path)
        first_file = next(folder_path.iterdir(), None)
        if first_file:
                model = TrainModule.load_from_checkpoint(first_file)
                self.model = model
        else:
            print("not exists model")

        return models