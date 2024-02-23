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

class EncoderDataset(Dataset):
        def __init__(self, data_array: torch.Tensor):
            self.

        def __len__(self):
            return len(self.data_array)

        def __getitem__(self, idx):
            sample = self.data_array[idx]
            return sample

class VariablePredictor:
    def __init__(self, model_path: str, batch_size: int):
        self.model_path = model_path
        self.batch_size = batch_size
        self.load_models(model_path)
    
    def load_models(self, model_path):
        folder_path = Path(model_path)
        first_file = next(folder_path.iterdir(), None)
        if first_file:
                model = TrainModule.load_from_checkpoint(first_file)
                self.model = model
        else:
            raise Exception("Not exists VariableEncoder model")
    
    def predict(self, dataset: torch.Tensor) -> torch.Tensor:
         