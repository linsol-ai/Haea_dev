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
from scipy import interpolate


def normalize_tensor(data):
    # 최솟값을 0으로 조정
    min_value = data.min(dim=-2, keepdim=True)[0]
    max_value = data.max(dim=-2, keepdim=True)[0]
    
    # 정규화
    output = (data - min_value) / (max_value - min_value)

    has_nan = torch.isnan(output).any()
    if has_nan:
        nan_indices = torch.isnan(output)
        output[nan_indices] = 0
    
    return output


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
        return torch.stack([wind_speed, sin_encoded, cos_encoded], dim=0)


def remove_missing_values(data):
    batch, width, height = data.shape
    interpolated_array = np.zeros_like(data)
    for i in range(batch):
        has_nan = np.isnan(data[i]).any()
        if has_nan:
            x = np.arange(width)
            y = np.arange(height)
            #mask invalid values
            array = np.ma.masked_invalid(data[i])
            xx, yy = np.meshgrid(x, y)
            #get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]

            GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                    (xx, yy),
                                        method='cubic')
            interpolated_array[i] = GD1
        else:
            interpolated_array[i] = data[i]
            
    return interpolated_array
        



class WeatherDataset:

    HAS_LEVEL_VARIABLE = [ 'geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

    NONE_LEVEL_VARIABLE = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'sea_surface_temperature', 'total_cloud_cover', 'total_precipitation_24hr', 'total_precipitation_6hr']

    HAS_LEVEL_WIND_VAR = ['u_component_of_wind', 'v_component_of_wind']

    NONE_LEVEL_WIND_VAR = ['10m_u_component_of_wind', '10m_v_component_of_wind']

    DATE_OFFSET = [(2021, 2016), (2016, 2011), (2013, 2008), (2008, 2003)]


    def __init__(self, year_offset:int, device:torch.device):
        end, start = self.DATE_OFFSET[year_offset]
        self.start = pd.to_datetime(f'{start}-12-31')
        self.end = pd.to_datetime(f'{end}-12-31')
        self.device = device

        dataset_urls = [
            f'gs://era5_preprocess/1440x720/{start}-12-31_{end}-12-31.zarr',
            f'gs://era5_preprocess/240x120/{start}-12-31_{end}-12-31.zarr',
            f'gs://era5_preprocess/60x30/{start}-12-31_{end}-12-31.zarr'
        ]
        self.load_dataset(dataset_urls)

    
    def load_dataset(self, dataset_urls):
        print("데이터셋 불러오는 중...")
        self.datasets = []
        for urls in dataset_urls:
            ds, _ = xb.open_zarr(urls)
            self.datasets.append(ds)


    def load_variable(self, dataset, key):
        data = dataset[key]
        data = data.to_numpy()
        if len(data.shape) == 4:
            removed = np.zeros_like(data)
            for i in range(data.size(0)):
                removed[i] = remove_missing_values(data[i])
            data = removed
        else:
            data = remove_missing_values(data)

        data = torch.from_numpy(data)
        # data.shape = (time, width, height)
        # data.shape = (time, width * height)

        if len(data.shape) == 4:
             data = data.flatten(2)
        else:
            data = data.flatten(1)
            
        return data
    

    def load(self):
        var_dataset = []
        wind_dataset = []

        for dataset in self.datasets:
            result = self.load_data(dataset)
            var_dataset.append(result[0])
            wind_dataset.append(result[1])
        
        # var_dataset.shape = (time, var * level, h * w)
        var_dataset = torch.stack(var_dataset, dim=2)
        wind_dataset = torch.stack(wind_dataset, dim=2)

        return var_dataset, wind_dataset

    
    def load_data(self, dataset:xr.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        start = time.time()
        result = {}

        print(dataset)

        with ThreadPoolExecutor() as executor:
            futures = {}

            for val in (self.NONE_LEVEL_VARIABLE + self.HAS_LEVEL_VARIABLE):
                print(val)
                key = executor.submit(self.load_variable, dataset, val)
                futures[key] = val

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val = futures[future]
                print(val)
                # shape => (time, level, h * w) or (time, h * w)
                data = future.result()
                if len(data.shape) == 3:
                    data = data.swapaxes(0, 1)
                result[val] = data



    def calculate_wind(self, u_wind, v_wind, device):
        res = preprocess_wind_data(u_wind, v_wind, device).cpu()
        return res



if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(0, device=device)
    weather.load()