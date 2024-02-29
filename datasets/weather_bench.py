import numpy as np
from tqdm import tqdm
from typing import Tuple
import numpy as np
import xarray_beam as xb
import xarray as xr
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
import sys,os
from pathlib import Path
import apache_beam as beam
import gc
from datetime import datetime

def normalize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std()
    # 정규화
    normalized_tensor = (tensor - mean) / std
    
    return normalized_tensor, mean, std


def calculate_wind_speed(u, v):
        return torch.sqrt(u**2 + v**2)
    

def calculate_wind_direction(u, v):
        return torch.arctan2(v, u) * (180 / torch.pi)
    

def cyclic_encoding(angle):
        """
        주어진 각도(angle)를 주기 코딩하여 사인과 코사인 값을 반환합니다.
        
        Parameters:
        - angle: 변환할 각도 (라디안 단위)
        
        Returns:
        - sin_val: 사인 값
        - cos_val: 코사인 값
        """
        sin_val = torch.sin(angle)
        cos_val = torch.cos(angle)
        return sin_val, cos_val


def preprocess_wind_data(u, v, device):
        u = u.to(device)
        v = v.to(device)

        # 1. 풍속 및 풍향 계산
        wind_speed = calculate_wind_speed(u, v)
        wind_direction = calculate_wind_direction(u, v)

        # 2. 풍향 주기 코딩
        sin_encoded, cos_encoded = cyclic_encoding(torch.deg2rad(wind_direction))

        wind_speed = wind_speed.cpu()
        wind_direction = wind_direction.cpu()
        sin_encoded = sin_encoded.cpu()
        cos_encoded = cos_encoded.cpu()

        un_wind_speed = wind_speed.clone().detach()
        un_sin_encoded = sin_encoded.clone().detach()
        un_cos_encoded = cos_encoded.clone().detach()

        wind_speed, wind_min, wind_max = normalize_tensor(wind_speed)
        sin_encoded, sin_min, sin_max  = normalize_tensor(sin_encoded)
        cos_encoded, cos_min, cos_max = normalize_tensor(cos_encoded)
        
        
        u = u.cpu()
        v = v.cpu()

        del u
        del v
        del wind_direction

        return torch.stack([wind_speed, sin_encoded, cos_encoded], dim=0), torch.stack([un_wind_speed, un_sin_encoded, un_cos_encoded], dim=0), (wind_min, sin_min, cos_min), (wind_max, sin_max, cos_max)


def download_zarr(source, output_path, variables=None):
    source_dataset, source_chunks = xb.open_zarr(source)
    if variables is not None:
        source_dataset = source_dataset[variables]

    template = (
      xb.make_template(source_dataset)
    )
    with beam.Pipeline() as root :
        (
            root
            | "Read from Source Dataset" >> xb.DatasetToChunks(source_dataset, source_chunks)
            | "Write to Zarr" >> xb.ChunksToZarr(output_path, template, source_chunks)
        )
        

class WeatherDataset:

    AIR_VARIABLE = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

    SURFACE_VARIABLE = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_cloud_cover', 'total_precipitation', 'total_column_water', 'total_column_water_vapour']

    VARIABLES = AIR_VARIABLE + SURFACE_VARIABLE

    DATE_OFFSET = [(2013, 2023), (2018, 2017), (2011, 2006), (2006, 2001)]

    GCS_BUCKET = 'gcs://era5_climate'

    RESOLUTION = ['512x256']

    PRESSURE_LEVEL = 37

    DIR_NAME = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'resource')


    def __init__(self, start_date: datetime, end_date: datetime, device: torch.device, normalize=True, offline=True):
        self.start_date = start_date
        self.end_date= end_date
        self.offline = offline
        self.device = device
        self.normalize = normalize
        dataset_path = self.check_dataset(start_date, end_date)
        self.load_dataset(dataset_path)
    

    def get_var_code(self, air_var, surface_var, level=37):
        code = []
        for air in air_var:
            idx = self.AIR_VARIABLE.index(air) + 1
            air_list = [ ((idx-1)*level) + i for i in range(level)]
            code.extend(air_list)

        for sur in surface_var:
            idx = len(self.AIR_VARIABLE) * level + self.SURFACE_VARIABLE.index(sur)
            code.append(idx)

        return code


    def check_dataset(self, start_date: datetime, end_date: datetime):
        idx = -1
        for i, offset in enumerate(self.DATE_OFFSET):
            start = datetime.strptime(f'{offset[0]}-01-01', '%Y-%m-%d')
            end = datetime.strptime(f'{offset[1]}-01-01', '%Y-%m-%d')
            if start <= start_date and end_date <= end:
                idx = i
                break
        
        if idx == -1:
            

            
        file_name = f'{start}-01-01_{end}-01-01.zarr'
        dataset_path = []

        for resol in self.RESOLUTION:
            if self.offline:
                folder = Path(self.DIR_NAME)
                file_path = folder / resol / file_name
                dataset_path.append(file_path)
                if not file_path.exists():
                    print("======= DOWNLOAD Zarr FROM GCS ======")
                    gcs_path = self.GCS_BUCKET + "/" + resol + "/" + file_name
                    print("DOWNLOAD: ", gcs_path)
                    download_zarr(gcs_path, file_path, self.VARIABLES)
            else:
                gcs_path = self.GCS_BUCKET + "/" + resol + "/" + file_name
                dataset_path.append(file_path)
        
        return dataset_path
    

    def load_dataset(self, dataset_path):
        print("데이터셋 불러오는 중...")
        self.datasets = []
        for path in dataset_path:
            ds = xr.open_zarr(path)
            self.datasets.append(ds)


    def load(self, air_variable, surface_variable, only_input_variable=[], constant_variables=[]):
        variables = air_variable + surface_variable + only_input_variable
        source_t, mean_std = self.load_data_single(self.datasets[0], variables, constant_variables)
        source_b, _ = self.load_data_single(self.datasets[1], variables, constant_variables)
        
        # var_dataset.shape = (time, var * level, h * w)
        source = torch.cat([source_t, source_b], dim=2)
        offset = len(only_input_variable)
        if offset > 0:
            mean_std = mean_std[:-len(only_input_variable), :]

        return source, source_t, mean_std
    
    def load_one(self, air_variable, surface_variable, only_input_variable=[], constant_variables=[]):
        variables = air_variable + surface_variable + only_input_variable
        source, mean_std = self.load_data_single(self.datasets[0], start_date, end_date, variables, constant_variables)
        
        offset = len(only_input_variable)
        if offset > 0:
            mean_std = mean_std[:-len(only_input_variable), :]

        return source, mean_std


    def load_variable_optimized(self, data: xr.DataArray):
        source = torch.from_numpy(data.values)  # `to_numpy()` 대신 `values` 사용
        if len(source.shape) == 4:
            stats = torch.empty((source.shape[1], 2), dtype=torch.float32)  # means와 stds를 담을 텐서 생성

            for i in range(source.size(1)):
                input, mean, std = normalize_tensor(source[:, i, :, :])
                source[:, i, :, :] = input
                stats[i] = torch.tensor([mean.item(), std.item()])

            return source.permute(1, 0, 2, 3).flatten(2), stats

        if len(source.shape) == 2:
            source, mean, std = normalize_tensor(source)
            return source.flatten(0), torch.tensor([mean, std])

        else:
            source, mean, std = normalize_tensor(source)
            return source.flatten(1), torch.tensor([mean, std])
        
    def load_data_single(self, dataset: xr.Dataset, start_date, end_date, variables, constant_variables):
        start = time.time()

        if start_date is not None:
            dataset = dataset.sel(time=slice(start_date, end_date))

        print("==== LOAD DATASET ====\n", dataset)

        result = [self.load_variable_optimized(dataset[val]) for val in tqdm(variables)]            

        # dataset.shape => (var*level, time, h * w)
        input_dataset = []
        mean_std_dataset = []

        for input, mean_std in result:
            if len(input.shape) == 3:
                for i in range(input.size(0)):
                    input_dataset.append(input[i].unsqueeze(0))
                    mean_std_dataset.append(mean_std[i].unsqueeze(0))
            else:
                input_dataset.append(input.unsqueeze(0))
                mean_std_dataset.append(mean_std.unsqueeze(0))

        input_dataset = torch.cat(input_dataset, dim=0)
        mean_std_dataset = torch.cat(mean_std_dataset, dim=0)

        if len(constant_variables) > 0:
            constant_dataset = []
            for val in constant_variables:
                input, _ = self.load_variable_optimized(dataset[val])
                constant_dataset.append(input.unsqueeze(0))

            # shape = (var, hidden)
            constant_dataset = torch.cat(constant_dataset, dim=0)
            # shape = (var, 1, hidden)
            constant_dataset = constant_dataset.unsqueeze(1)
            constant_dataset = constant_dataset.repeat(1, input_dataset.size(1), 1)
            input_dataset = torch.cat([input_dataset, constant_dataset], dim=0)

        # dataset.shape => (time, var, h * w)
        input_dataset = input_dataset.swapaxes(0, 1)
        
        end = time.time()
        print(f"{end - start:.5f} sec")
        return input_dataset, mean_std_dataset


    def load_data(self, dataset:xr.Dataset, variables, constant_variables) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = time.time()
        result = {}

        print("==== LOAD DATASET ====\n", dataset)

        with ThreadPoolExecutor() as executor:
            futures = {}

            for val in variables:
                key = executor.submit(self.load_variable_optimized, dataset[val])
                futures[key] = val

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val = futures[future]
                # shape => (level, time, h * w) or (time, h * w)
                input, mean_std = future.result()
                result[val] = (input, mean_std)
            

        # dataset.shape => (var*level, time, h * w)
        input_dataset = []
        mean_std_dataset = []

        for val in variables:
            input, mean_std = result[val]
            if len(input.shape) == 3:
                for i in range(input.size(0)):
                    input_dataset.append(input[i].unsqueeze(0))
                mean_std_dataset.append(mean_std.swapaxes(0, 1))
            else:
                input_dataset.append(input.unsqueeze(0))
                mean_std_dataset.append(mean_std.unsqueeze(0))

        input_dataset = torch.cat(input_dataset, dim=0)
        mean_std_dataset = torch.cat(mean_std_dataset, dim=0)

        if len(constant_variables) > 0:
            constant_dataset = []
            for val in constant_variables:
                input, mean_std = self.load_variable_optimized(dataset[val])
                constant_dataset.append(input.unsqueeze(0))

            # shape = (var, hidden)
            constant_dataset = torch.cat(constant_dataset, dim=0)
            # shape = (var, 1, hidden)
            constant_dataset = constant_dataset.unsqueeze(1)
            print(constant_dataset.shape)
            constant_dataset = constant_dataset.repeat(1, input_dataset.size(1), 1)
            input_dataset = torch.cat([input_dataset, constant_dataset], dim=0)

        # dataset.shape => (time, var, h * w)
        input_dataset = torch.swapaxes(input_dataset, 0, 1)
        
        end = time.time()
        print(f"{end - start:.5f} sec")
        return input_dataset, mean_std_dataset


    def calculate_wind(self, u_wind, v_wind, device):
        res = preprocess_wind_data(u_wind, v_wind, device)
        torch.cuda.empty_cache()
        return res


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(0, device=device, offline=True)
    print(weather.load_one(weather.AIR_VARIABLE, weather.SURFACE_VARIABLE)[0].shape)