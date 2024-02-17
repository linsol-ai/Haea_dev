import argparse
import logging
import tqdm
import pytorch_lightning as pl
import torch
from typing import Tuple
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

class VariableProprecessor:
    def __init__(self, model_path: str, year_offset: int, latitude:Tuple, longitude, variables=WeatherDataset.HAS_LEVEL_VARIABLE+WeatherDataset.NONE_LEVEL_VARIABLE, batch_size = 512):
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

    def load_models(self, variables, model_path):
        models = {}
        for key in variables:
            folder_path = Path(os.path.join(model_path, key))
            first_file = next(folder_path.iterdir(), None)
            if first_file:
                print(f"====== LOAD MODELS : {key} =======")
                model = DVAETrainModule.load_from_checkpoint(first_file)
                models[key] = model
            else:
                print("변수 폴더가 비어있습니다.")
                break

        return models


    def predict_vars(self, key: str, model: DVAETrainModule, dataset: torch.Tensor):
        print(f"====== PREDICT : {key} =======")
        shape = dataset.shape
        dataset = ImageDataset(dataset)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=8, shuffle=False
        )
        predictions = []
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            # shape = (batch, hidden_dim)
            predict = model(batch.to(self.device)).cpu()
            predictions.append(predict)
        
        predictions = torch.cat(predictions, dim=0)

        if len(shape) == 5:
            predictions = predictions.view(shape[0], shape[1], predictions.size(1))
        else:
            predictions = predictions.unsqueeze(0)

        print("RESULT SHAPE: ", predictions.shape)
        return predictions


    def predict(self) -> torch.Tensor:
        source_dataset = []
        target_dataset = []
        mean_std_set = []

        for key in self.variables:
            source = self.input[key]
            target = self.target[key]
            mean_std = self.normalizaion[key]
            model = self.models[key]
            shape = source.shape
            # predict.shape = (levels, time, hidden)
            predict = self.predict_vars(key, model, source)
            source_dataset.append(predict)

            if len(shape) < 5:
                target = target.unsqueeze(0)
                mean_std = mean_std.unsqueeze(0)
            else:
                mean_std = mean_std.swapaxes(0, 1)
            
            print(mean_std.shape)
            target_dataset.append(target)
            mean_std_set.append(mean_std)
        
        # dataset.shape = (vars, time, hidden)
        source_dataset = torch.cat(source_dataset, dim=0)
        target_dataset = torch.cat(target_dataset, dim=0)
        # mean_std_set.shape = (vars, 2)
        mean_std_set = torch.cat(mean_std_set, dim=0)

        return source_dataset.swapaxes(0, 1), target_dataset.swapaxes(0, 1), mean_std_set
    


if __name__ == '__main__':
    processor = VariableProprecessor(WeatherDataset.HAS_LEVEL_VARIABLE + WeatherDataset.NONE_LEVEL_VARIABLE, '/workspace/Haea_dev/checkpoints/dVAE', 0)
    source, target, mean_std = processor.predict()
    print(source.shape)
    print(target.shape)
    print(mean_std.shape)