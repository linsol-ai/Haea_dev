import numpy as np
from tqdm import tqdm

import numpy as np
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

        wind_speed = normalize_tensor(wind_speed)
        sin_encoded = normalize_tensor(sin_encoded)
        cos_encoded = normalize_tensor(cos_encoded)
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


    def __init__(self, start:int, end:int, device:torch.device):
        self.start = pd.to_datetime(f'{start}-01-01')
        self.end = pd.to_datetime(f'{end}-01-01')
        self.device = device
        
        dataset_urls = [
            f'gs://era5_preprocess/1440x720/{start}-01-01_{end}-01-01.zarr',
            f'gs://era5_preprocess/240x121/{start}-01-01_{end}-01-01.zarr',
            f'gs://era5_preprocess/64x32/{start}-01-01_{end}-01-01.zarr'
        ]
        self.load_dataset(dataset_urls)

    
    def load_dataset(self, dataset_urls):
        print("데이터셋 불러오는 중...")
        self.datasets = []
        for urls in dataset_urls:
            self.datasets.append(xr.open_zarr(urls, chunks=None, consolidated=True))


    def load_level_val(self, key, level):
        arr = self.ds[key]
        data = arr.sel(level=level)
        data = data.to_numpy()
        data = remove_missing_values(data)
        data = torch.from_numpy(data)
        return data


    def load_level_data(self, ):
         

    
    
    def load_data(self, variables, start_date, end_date, wind_batch, device):
        wind_keys = ['u_component_of_wind', 'v_component_of_wind']
        levels = self.ds.level.values
        result = {}
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = {}
            for val in variables:
                result[val] = {}
                for level in levels:
                    key = executor.submit(self.load_level_val, val, level, start_date, end_date, val not in wind_keys)
                    futures[key] = (val, level)

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val, level = futures[future]
                # shape => (level, time, h, w)
                data = future.result()
                result[val][level] = data
        
        wind_result = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            for level in levels:
                u_wind = result[wind_keys[0]][level]
                v_wind = result[wind_keys[1]][level]
                key = executor.submit(self.calculate_wind, u_wind, v_wind, wind_batch, device)
                futures[key] = level

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                level = futures[future]
                # shape => (3, time, h, w)
                data = future.result()
                wind_result[level] = torch.swapaxes(data, 0, 1)

            del result[wind_keys[0]]
            del result[wind_keys[1]]


        # shape => (var*level, time, h, w)
        dataset = []
        for val in variables:
            if val in wind_keys:
                continue
            for level in levels:
                data = result[val][level]
                dataset.append(data) 
        
        # shape => (level, time, 3, h, w)
        wind_dataset = []
        for level in levels:
            wind_dataset.append(wind_result[level])

        dataset = torch.stack(dataset, dim=0)
        # shape => (time, var, h, w)
        dataset = torch.swapaxes(dataset, 0, 1)
        dataset = torch.unsqueeze(dataset, dim=2)
        # shape => (time, var, level, c, h, w)
        print(dataset.shape)

        wind_dataset = torch.stack(wind_dataset, dim=0)
        # shape => (time, level, 3, h, w)
        wind_dataset = torch.swapaxes(wind_dataset, 0, 1)

        end = time.time()
        print(f"{end - start:.5f} sec")
        return dataset, wind_dataset


    def calculate_wind(self, u_wind, v_wind, batch, device):
        part_size = (len(u_wind) // batch) + 1
        
        u_wind = torch.chunk(torch.from_numpy(u_wind), part_size, 0)
        v_wind = torch.chunk(torch.from_numpy(v_wind), part_size, 0)
        output = []

        for b in range(part_size):
            res = preprocess_wind_data(u_wind[b], v_wind[b], device).cpu()
            output.append(res)

        return torch.concat(output, dim=1)



if __name__ == '__main__':
    weather = WeatherDataset(url='gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
    weather.load_init()

    variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-02-01')
    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)


    output = weather.load_bart(variable, levels, start_date, end_date, (lat_min, lat_max), (lon_min, lon_max), 128, device)