import numpy as np
from tqdm import tqdm

import numpy as np
import xarray
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch

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

def preprocess_wind_data(u, v, device, normalize, flatten):
        u = u.to(device)
        v = v.to(device)

        # 1. 풍속 및 풍향 계산
        wind_speed = calculate_wind_speed(u, v)
        wind_direction = calculate_wind_direction(u, v)

        # 2. 풍향 주기 코딩
        sin_encoded, cos_encoded = cyclic_encoding(torch.deg2rad(wind_direction))
        
        if normalize:
            wind_speed = normalize_tensor(wind_speed)
            sin_encoded = normalize_tensor(sin_encoded)
            cos_encoded = normalize_tensor(cos_encoded)
        
        if flatten:
            wind_speed = remove_nan(wind_speed.flatten())
            sin_encoded = remove_nan(sin_encoded.flatten())
            cos_encoded = remove_nan(cos_encoded.flatten())

        return torch.stack([wind_speed, sin_encoded, cos_encoded], dim=0)


def remove_nan(input:torch.Tensor):
    has_nan = torch.isnan(input).any()
    if has_nan:
        nan_indices = torch.isnan(input)
        input[nan_indices] = input.mean(dim=-1)
    return input


class WeatherDataset:

    def __init__(self, url):
        self.url = url
    

    def load_init(self, start_date, end_date, lat, lon):
        print("데이터셋 불러오는 중...")
        ds = xarray.open_zarr(self.url, chunks="auto")

        ds = ds.sel(time=slice(start_date, end_date))

        lat_min, lat_max = lat
        lon_min, lon_max = lon

        # 해당 범위에 속하는 위도와 경도의 인덱스 찾기
        lat_indices = np.where((ds.latitude >= lat_min) & (ds.latitude <= lat_max))[0]
        lon_indices = np.where((ds.longitude >= lon_min) & (ds.longitude <= lon_max))[0]

        self.ds = ds.isel(latitude=lat_indices, longitude=lon_indices)
        print(self.ds)


    def get_key_without_level(self):
        variables_without_level = [var for var in self.ds.data_vars if 'level' not in self.ds[var].dims]
        return variables_without_level


    def load_data_unet(self, key, level, start_date, end_date, normalize=True):
        arr = self.ds[key]
        arr = arr.sel(time=slice(start_date, end_date))
        data = arr.sel(level=level)
        data = data.to_numpy()
        data = torch.from_numpy(data)
        if normalize:
             data = normalize_tensor(data)
        return data


    def load_val(self, key):
        arr = self.ds[key]
        data = arr.to_numpy()
        data = torch.from_numpy(data)
        data = data.transpose(0, )
        data = remove_nan(data.flatten(1))
        print(data.shape)
        return data
    

    def load_bart(self, variables, levels, wind_batch, device):
        wind_keys = ['u_component_of_wind', 'v_component_of_wind']
        result = {}
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = {}
            for val in variables:
                key = executor.submit(self.load_val, val)
                futures[key] = val

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val = futures[future]
                # shape => (level, time, h * w)
                data = future.result()
                print(data.shape)
                result[val] = data
        
        wind_result = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            u_wind = result[wind_keys[0]]
            v_wind = result[wind_keys[1]]
            key = executor.submit(self.calculate_wind, u_wind, v_wind, wind_batch, device, False, True)
            futures[key] = level

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                level = futures[future]
                # shape => (3, time, h * w)
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

    
   
        wind_keys = ['u_component_of_wind', 'v_component_of_wind']

        result = {}
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = {}
            for val in variables:
                result[val] = {}
                for level in levels:
                    key = executor.submit(self.load_level_val, val, level, val not in wind_keys)
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
                key = executor.submit(self.calculate_wind, u_wind, v_wind, wind_batch, device, True, False)
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




    def load_unet(self, variable, levels, start_date, end_date, normalize=True):
        result = {}
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.load_data_unet, variable, level, start_date, end_date, normalize): level for level in levels}
            for future in tqdm(as_completed(futures), desc="Processing futures"):
                key = futures[future]
                # shape => (time, h, w)
                result[key] = future.result()

        # shape => (level, time, h, w)
        dataset = []
        for key in levels:
            dataset.append(result[key])

        dataset = torch.stack(dataset, dim=0)
        dataset = dataset.reshape(-1, dataset.size(2), dataset.size(3))
        dataset = torch.unsqueeze(dataset, dim=1)
        # shape => (time, level, c, h, w)
        print(dataset.shape)
        end = time.time()
        print(f"{end - start:.5f} sec")
        return dataset


    def calculate_wind(self, u_wind, v_wind, batch, device, normalize=True, flatten=False):
        part_size = (len(u_wind) // batch) + 1
        
        # (level, time, h * w)
        u_wind = torch.chunk(u_wind, part_size, 0)
        v_wind = torch.chunk(v_wind, part_size, 0)
        output = []

        for b in range(part_size):
            res = preprocess_wind_data(u_wind[b], v_wind[b], device, normalize, flatten).cpu()
            output.append(res)

        return torch.concat(output, dim=1)


    def load_data_wind(self, level, start_date, end_date, batch, device):
        keys = ['u_component_of_wind', 'v_component_of_wind']
        result = []

        for i, key in enumerate(keys):
            arr = self.ds[key]

            if 'time' in arr.dims:
                arr = arr.sel(time=slice(start_date, end_date))
                data = arr.sel(level=level)
                data = data.to_numpy()
                result.append(torch.from_numpy(data))

        result[0] = torch.chunk(result[0], batch, 0)
        result[1] = torch.chunk(result[1], batch, 0)
        output = []

        for b in range(batch):
            res = preprocess_wind_data(result[0][b], result[1][b], device).cpu()
            # torch.Size([3, 13, 512 * 256])
            output.append(res)

        return torch.concat(output, dim=1)


    def load_wind(self, levels, start_date, end_date, batch, device):
        result = {}
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.load_data_wind, level, start_date, end_date, batch, device): level for level in levels}
            for future in tqdm(as_completed(futures), desc="Processing futures"):
                key = futures[future]
                # shape => (3, time, h, w)
                result[key] = future.result()
                # shape => (time, 3, h, w)
                result[key] = torch.swapaxes(result[key], 0, 1)

        # shape => (level, time, 3, h, w)
        dataset = []
        for key in levels:
            dataset.append(result[key])

        dataset = torch.stack(dataset, dim=0)
        # shape => (time, level, 3, h, w)
        dataset = torch.swapaxes(dataset, 0, 1)

        print(dataset.shape)
        end = time.time()
        print(f"{end - start:.5f} sec")
        return dataset


if __name__ == '__main__':
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-02-01')

    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131

    weather = WeatherDataset(url='gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
    weather.load_init(start_date, end_date, (lat_min, lat_max), (lon_min, lon_max))
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
    
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    output = weather.load_bart(variable, levels, 256, device)