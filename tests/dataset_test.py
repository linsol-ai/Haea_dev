import argparse
import sys
import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from torchvision import utils

from tqdm import tqdm

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from datasets.weather_bench import WeatherDataset



class ImageDataset(Dataset):
        def __init__(self, data_array):
            self.data_array = data_array

        def __len__(self):
            return len(self.data_array)

        def __getitem__(self, idx):
            sample = self.data_array[idx]
            return sample
        

def main(args):
    device = "cuda"

    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    weather = WeatherDataset(url='gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr')
    weather.load_init()

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-09-01')


if __name__ == "__main__":
    variable_keys = ['geopotential', 'specific_humidity', 'temperature', 'vertical_velocity']

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str, default="cycle")
    parser.add_argument("--key", type=str, default=variable_keys[0])

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
