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


def download_zarr(source, output_path):
    source_dataset, source_chunks = xb.open_zarr(source)

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

    HAS_LEVEL_VARIABLE = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

    NONE_LEVEL_VARIABLE = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_cloud_cover', 'total_precipitation']

    PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    HAS_LEVEL_WIND_VAR = ['u_component_of_wind', 'v_component_of_wind']

    NONE_LEVEL_WIND_VAR = ['10m_u_component_of_wind', '10m_v_component_of_wind']

    DATE_OFFSET = [(2021, 2016), (2019, 2017), (2011, 2006), (2006, 2001)]

    GCS_BUCKET = 'gcs://era5_climate'

    RESOLUTION = ['1440x720']

    RESOLUTION_MODE_FULL_SET = 3
    RESOLUTION_MODE_LARGE_SET = 2
    RESOLUTION_MODE_BASIC_SET = 1

    DIR_NAME = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'resource')


    def __init__(self, year_offset: int, device: torch.device, normalize=True, offline = True):
        end, start = self.DATE_OFFSET[year_offset]
        self.start = pd.to_datetime(f'{start}-12-31')
        self.end = pd.to_datetime(f'{end}-12-31')
        self.offline = offline
        self.device = device
        self.normalize = normalize
        dataset_path = self.check_dataset(start, end)
        self.load_dataset(dataset_path)

    
    def check_dataset(self, start, end):
        file_name = f'{start}-12-31_{end}-12-31.zarr'
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
                    download_zarr(gcs_path, file_path)
            else:
                gcs_path = self.GCS_BUCKET + "/" + resol + "/" + file_name
                dataset_path.append(file_path)
        
        return dataset_path
    

    def load_dataset(self, dataset_path):
        print("데이터셋 불러오는 중...")
        self.datasets = []
        for path in dataset_path:
            ds, _ = xb.open_zarr(path)
            self.datasets.append(ds)



    def load_variable_2D(self, data: xr.DataArray, key, lat_indices = None, lon_indices = None):
        source = data.to_numpy()
        source = torch.from_numpy(source)
        # data.shape = (time, width, height)
        # or data.shape = (time, level, width, height)

        has_nan = torch.isnan(source).any()

        if has_nan:
            print('====== nan warning =======')
            print("key: ", key)
            nan_indices = torch.isnan(source)
            source[nan_indices] = 0

        target = source.clone().detach()

        if len(source.shape) == 4:
            if lat_indices is not None:
                target = target[:, :, lat_indices, :][:, :, :, lon_indices]

            inputs = []
            means = []
            stds = []
            for i in range(source.size(1)):
                input, mean, std = normalize_tensor(source[:, i, : , :])
                inputs.append(input)
                means.append(mean)
                stds.append(std)

            return torch.stack(inputs, dim=0).unsqueeze(2), target.swapaxes(0, 1).flatten(2), torch.tensor([means, stds])

        else:
            if lat_indices is not None:
                target = target[:, lat_indices, :][:, :, lon_indices]

            input, mean, std = normalize_tensor(source)
            return input.unsqueeze(1), target.flatten(1), torch.tensor([mean, std])


    def load_variable(self, data: xr.DataArray, key):
        source = data.to_numpy()
        source = torch.from_numpy(source)
        # data.shape = (time, width, height)
        # or data.shape = (time, level, width, height)
        target = source.clone().detach()

        if len(source.shape) == 4:
            inputs = []
            means = []
            stds = []
            for i in range(source.size(1)):
                input, mean, std = normalize_tensor(source[:, i, : , :])
                inputs.append(input)
                means.append(mean)
                stds.append(std)

            return torch.stack(inputs, dim=0).flatten(2), target.swapaxes(0, 1).flatten(2), torch.tensor([means, stds])

        else:
            input, mean, std = normalize_tensor(source)
            return input.flatten(1), target.flatten(1), torch.tensor([mean, std])
    

    def load_2D(self, variables = HAS_LEVEL_VARIABLE + NONE_LEVEL_VARIABLE, latitude: Tuple | None = None, longitude: Tuple | None = None):
        input, target, mean_std = self.load_data_2D(self.datasets, variables, latitude=latitude, longitude=longitude)
        return input, target, mean_std
    

    def load(self, variables = HAS_LEVEL_VARIABLE + NONE_LEVEL_VARIABLE):
        input, target, mean_std = self.load_data(self.datasets[0], variables)

        source_dataset = []
        target_dataset = []
        normalizaion_data = []

        for dataset in self.datasets:
            input, target, mean_std = self.load_data(dataset, variables)
            source_dataset.append(input)
            target_dataset.append(target)
            normalizaion_data.append(normalizaion)
        
        # var_dataset.shape = (time, var * level, h * w)
        input_dataset = torch.concat(input_dataset, dim=2)
        target_dataset = torch.concat(target_dataset, dim=2)
        normalizaion_data = torch.stack(normalizaion_data, dim=0)


        return input, target, mean_std



    def load_data_2D(self, dataset:xr.Dataset, variables, latitude: Tuple | None = None, longitude: Tuple | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = time.time()
        result = {}

        print("==== LOAD DATASET ====\n", dataset)
        lat_indices, lon_indices = (None, None)
        
        if latitude:
            lat_min, lat_max = latitude
            lon_min, lon_max = longitude

            lat_indices = np.where((dataset.latitude >= lat_min) & (dataset.latitude <= lat_max))[0]
            lon_indices = np.where((dataset.longitude >= lon_min) & (dataset.longitude <= lon_max))[0]


        with ThreadPoolExecutor() as executor:
            futures = {}

            for val in variables:
                key = executor.submit(self.load_variable_2D, dataset[val], val, lat_indices, lon_indices)
                futures[key] = val

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val = futures[future]
                # shape => (level, time, h, w) or (time, h, w)
                input, target, mean_std = future.result()
                result[val] = (input, target, mean_std)
            

        # dataset.shape => (var*level, time, h, w)
        input_dataset = {}
        target_dataset = {}
        mean_std_dataset = {}

        for val in variables:
            input, target, mean_std = result[val]
            input_dataset[val] = input
            target_dataset[val] = target
            mean_std_dataset[val] = mean_std
        
        end = time.time()
        print(f"{end - start:.5f} sec")
        return input_dataset, target_dataset, mean_std_dataset
    

    def load_data(self, dataset:xr.Dataset, variables) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = time.time()
        result = {}

        print("==== LOAD DATASET ====\n", dataset)

        with ThreadPoolExecutor() as executor:
            futures = {}

            for val in variables:
                key = executor.submit(self.load_variable, dataset[val], val)
                futures[key] = val

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val = futures[future]
                # shape => (level, time, h * w) or (time, h * w)
                input, target, mean_std = future.result()
                result[val] = (input, target, mean_std)
            

        # dataset.shape => (var*level, time, h * w)
        input_dataset = []
        target_dataset = []
        mean_std_dataset = []

        for val in variables:
            input, target, mean_std = result[val]
            if len(input.shape) == 3:
                for i in range(input.size(0)):
                    input_dataset.append(input[i])
                    target_dataset.append(target[i])
                mean_std_dataset.append(mean_std.swapaxes(0, 1))
            else:
                input_dataset.append(input)
                target_dataset.append(target)
                mean_std_dataset.append(mean_std.unsqueeze(0))


        input_dataset = torch.stack(input_dataset, dim=0)
        target_dataset = torch.stack(target_dataset, dim=0)
        mean_std_dataset = torch.cat(mean_std_dataset, dim=0)

        # dataset.shape => (time, var, h * w)
        input_dataset = torch.swapaxes(input_dataset, 0, 1)
        target_dataset = torch.swapaxes(target_dataset, 0, 1)
        
        end = time.time()
        print(f"{end - start:.5f} sec")
        return input_dataset, target_dataset, mean_std_dataset


    def calculate_wind(self, u_wind, v_wind, device):
        res = preprocess_wind_data(u_wind, v_wind, device)
        torch.cuda.empty_cache()
        return res


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(0, device=device, offline=True)
    lat = (32, 39.3)
    lon = (124, 131.4)
    print(weather.load(weather.HAS_LEVEL_VARIABLE+weather.NONE_LEVEL_VARIABLE)[0].shape)