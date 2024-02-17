import argparse
import logging

import pytorch_lightning as pl
import torch
from typing import Dict
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from datasets.weather_bench import WeatherDataset
from models.dVAE.training.lightning import DVAETrainModule


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
    def __init__(self, variables, model_path: str, year_offset: int, batch_size = 256):
        self.variables = variables
        self.model_path = model_path
        self.batch_size = batch_size
        self.input, self.target, self.normalizaion = self.load_dataset(variables, year_offset)
        self.models = self.load_models(variables, model_path)

    def load_dataset(self, variables, year_offset):
        device = ("cuda" if torch.cuda.is_available() else "cpu" )
        self.device = torch.device(device)
        weather = WeatherDataset(year_offset, device=device, offline=True)
        return weather.load(variables=variables)

    def load_models(self, variables, model_path) -> Dict[DVAETrainModule]:
        models = {}
        for key in variables:
            folder_path = Path(os.path.join(model_path, key))
            first_file = next(folder_path.iterdir(), None)
            if first_file:
                print(f"====== LOAD MODELS : {key} =======")
                model = DVAETrainModule.load_from_checkpoint(first_file)
                models[key] = model
            else:
                print("폴더가 비어있습니다.")
                break

        return models


    def predict_vars(self, key: str, model: DVAETrainModule, dataset: torch.Tensor):
        print(f"====== PREDICT : {key} =======")
        dataset = ImageDataset(dataset)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )
        trainer = pl.Trainer(accelerator="auto",)
        predictions = trainer.predict(model, data_loader)
        return predictions


    def predict(self):
        for key in self.variables:
            source_data = self.input[key]
            model = self.models[key]
            predictions = self.predict_vars(key, model, source_data)
            print(predictions)