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

    DATE_OFFSET = [(2021, 2019), (2019, 2017), (2011, 2006), (2006, 2001)]

    GCS_BUCKET = 'gcs://era5_climate'

    RESOLUTION = ['1440x720', '240x120', '60x30']

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
        dataset_path = None

        if self.offline:
            folder = Path(self.DIR_NAME)
            file_path = folder / self.RESOLUTION[0] / file_name
            dataset_path = file_path
            if not file_path.exists():
                print("======= DOWNLOAD Zarr FROM GCS ======")
                gcs_path = self.GCS_BUCKET + "/" + self.RESOLUTION[0] + "/" + file_name
                print("DOWNLOAD: ", gcs_path)
                download_zarr(gcs_path, file_path)
        else:
            gcs_path = self.GCS_BUCKET + "/" + self.RESOLUTION[0] + "/" + file_name
            dataset_path = file_path
        
        return dataset_path
    

    def load_dataset(self, dataset_path):
        print("데이터셋 불러오는 중...")
        ds, _ = xb.open_zarr(dataset_path)
        self.datasets = ds


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
    

    def load_1D(self, variables = HAS_LEVEL_VARIABLE + NONE_LEVEL_VARIABLE, latitude: Tuple | None = None, longitude: Tuple | None = None):
        input, target, mean_std = self.load_data_1D(self.datasets, variables, latitude=latitude, longitude=longitude)
        return input, target, mean_std



    """""
    def load_data(self, dataset:xr.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        start = time.time()
        result = {}

        print("==== LOAD DATASET ====\n", dataset)

        with ThreadPoolExecutor() as executor:
            futures = {}

            for val in (self.NONE_LEVEL_VARIABLE + self.HAS_LEVEL_VARIABLE):
                key = executor.submit(self.load_variable, dataset[val])
                futures[key] = val

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val = futures[future]
                # shape => (level, time, h * w) or (time, h * w)
                input, target, min, max = future.result()

                if len(input.shape) == 3:
                    input = input.swapaxes(0, 1)
                    target = target.swapaxes(0, 1)

                result[val] = (input, target, min, max)
            
        wind_result = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            v1 = result[self.HAS_LEVEL_WIND_VAR[0]], result[self.HAS_LEVEL_WIND_VAR[1]]
            k1 = executor.submit(self.calculate_wind, v1[0][1], v1[1][1], self.device)
            futures[k1] = 1

            v2 = result[self.NONE_LEVEL_WIND_VAR[0]], result[self.NONE_LEVEL_WIND_VAR[1]]
            k2 = executor.submit(self.calculate_wind, v2[0][1], v2[1][1], self.device)
            futures[k2] = 0

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                level = futures[future]
                # shape => (3, time, h * w) or (level * 3, time, h * w)
                input, target, min, max = future.result()
                if len(input.shape) == 4:
                    input = input.view(-1, input.size(2), input.size(3))
                    target = target.view(-1, target.size(2), target.size(3))

                wind_result[level] = (input, target, min, max)

            del result[self.HAS_LEVEL_WIND_VAR[0]]
            del result[self.HAS_LEVEL_WIND_VAR[1]]
            del result[self.NONE_LEVEL_WIND_VAR[0]]
            del result[self.NONE_LEVEL_WIND_VAR[1]]


        # dataset.shape => (var*level, time, h * w)
        input_dataset = []
        target_dataset = []
        min_max_data = []

        for val in (self.HAS_LEVEL_VARIABLE + self.NONE_LEVEL_VARIABLE):
            if val in (self.HAS_LEVEL_WIND_VAR + self.NONE_LEVEL_WIND_VAR):
                continue
            input, target, min, max = result[val]
            if len(input.shape) == 3:
                for i in range(input.size(0)):
                    input_dataset.append(input[i])
                    target_dataset.append(target[i])
                    min_max_data.append([min, max])
            else:
                input_dataset.append(input)
                target_dataset.append(target)
                min_max_data.append([min, max])


        input_dataset = torch.stack(input_dataset, dim=0)
        target_dataset = torch.stack(target_dataset, dim=0)

        # dataset.shape => (time, var, h * w)
        input_dataset = torch.swapaxes(input_dataset, 0, 1)
        target_dataset = torch.swapaxes(target_dataset, 0, 1)
        
        # wind.shape => (3, time, h * w)
        input_wind_dataset = []
        input_wind_dataset.extend(wind_result[0][0])
        min_max_data.append([wind_result[0][2][0], wind_result[0][3][0]])
        min_max_data.append([wind_result[0][2][1], wind_result[0][3][1]])
        min_max_data.append([wind_result[0][2][2], wind_result[0][3][2]])

        # wind.shape => (level * 3, time, h * w)
        input_wind_dataset.extend(wind_result[1][0])
        for i in range(wind_result[1][0].size(0) // 3):
            min_max_data.append([wind_result[1][2][0], wind_result[1][3][0]])
            min_max_data.append([wind_result[1][2][1], wind_result[1][3][1]])
            min_max_data.append([wind_result[1][2][2], wind_result[1][3][2]])

        target_wind_dataset = []
        target_wind_dataset.extend(wind_result[0][1])
        target_wind_dataset.extend(wind_result[1][1])

        # wind.shape => (level, time, h * w)
        input_wind_dataset = torch.stack(input_wind_dataset, dim=0)
        target_wind_dataset = torch.stack(target_wind_dataset, dim=0)
        min_max_data = torch.tensor(min_max_data)

        # shape => (time, level, h * w)
        input_wind_dataset = torch.swapaxes(input_wind_dataset, 0, 1)
        target_wind_dataset = torch.swapaxes(target_wind_dataset, 0, 1)
        
        end = time.time()
        print(f"{end - start:.5f} sec")
        return torch.concat([input_dataset, input_wind_dataset], dim=1), torch.concat([target_dataset, target_wind_dataset], dim=1), min_max_data
    """""

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
    

    def load_data_1D(self, dataset:xr.Dataset, variables, latitude: Tuple | None = None, longitude: Tuple | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                key = executor.submit(self.load_variable_1D, dataset[val], val, lat_indices, lon_indices)
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
    print(weather.load_1D(weather.HAS_LEVEL_VARIABLE+weather.NONE_LEVEL_VARIABLE, lat, lon)[0].shape)