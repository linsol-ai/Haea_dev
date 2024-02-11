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
from dask.diagnostics import ProgressBar
import apache_beam as beam

def normalize_tensor(tensor):
    # tensor는 (time, level, width, height)의 차원을 가진다고 가정
    shape = tensor.shape
    # 최대, 최소 스케일링을 수행하기 위해 (time, level) 기준으로 reshape
    tensor = tensor.view(shape[0], shape[1], -1)
    
    # 최대값과 최소값을 찾음
    min_val = tensor.min(dim=2, keepdim=True)[0]
    max_val = tensor.max(dim=2, keepdim=True)[0]
    
    # 분모가 0이 되는 것을 방지
    denom = max_val - min_val
    denom[denom == 0] = 1
    
    # 최대-최소 스케일링 수행
    scaled_tensor = (tensor - min_val) / denom
    
    # 원래 차원으로 되돌림
    scaled_tensor = scaled_tensor.view(shape)
    
    return scaled_tensor


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


def preprocess_wind_data(u, v, device, normalize):
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
        

        if normalize:
            wind_speed = normalize_tensor(wind_speed).cpu()
            sin_encoded = normalize_tensor(sin_encoded).cpu()
            cos_encoded = normalize_tensor(cos_encoded).cpu()
        
        u = u.cpu()
        v = v.cpu()

        del u
        del v
        del wind_direction

        return torch.stack([wind_speed, sin_encoded, cos_encoded], dim=0)


class ProgressUpdater(beam.DoFn):

    def __init__(self, pbar):
        self.pbar = pbar

    def process(self, element):
        # Update progress bar for each element processed
        self.update_progress(1)
        yield element
    
    def update_progress(self, progress):
        self.pbar.update(progress)


def download_zarr(source, output_path):
    source_dataset, source_chunks = xb.open_zarr(source)
    template = (
      xb.make_template(source_dataset)
    )
    pbar = tqdm(total=100)
    with beam.Pipeline() as root :
        (
            root
            | "Read from Source Dataset" >> xb.DatasetToChunks(source_dataset, source_chunks)
            | "Update Progress Bar" >> beam.ParDo(ProgressUpdater(pbar=pbar))
            | "Write to Zarr" >> xb.ChunksToZarr(output_path, template, source_chunks)
        )
        

class WeatherDataset:

    HAS_LEVEL_VARIABLE = [ 'geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

    NONE_LEVEL_VARIABLE = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_cloud_cover', 'total_precipitation_24hr', 'total_precipitation_6hr']

    HAS_LEVEL_WIND_VAR = ['u_component_of_wind', 'v_component_of_wind']

    NONE_LEVEL_WIND_VAR = ['10m_u_component_of_wind', '10m_v_component_of_wind']

    DATE_OFFSET = [(2021, 2016), (2016, 2011), (2011, 2006), (2006, 2001)]

    GCS_BUCKET = 'gcs://era5_preprocess'

    RESOLUTION = ['1440x720', '240x120', '60x30']

    DIR_NAME = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'resource')


    def __init__(self, year_offset:int, device:torch.device, normalize=True):
        end, start = self.DATE_OFFSET[year_offset]
        self.start = pd.to_datetime(f'{start}-12-31')
        self.end = pd.to_datetime(f'{end}-12-31')
        self.device = device
        self.normalize = normalize
        dataset_path = self.check_dataset(self.RESOLUTION, start, end)
        
        self.load_dataset(dataset_path)

    
    def check_dataset(self, resolutions, start, end):
        file_name = f'{start}-12-31_{end}-12-31.zarr'
        dataset_path = []

        for resol in resolutions:
            folder = Path(self.DIR_NAME)
            file_path = folder / resol / file_name
            dataset_path.append(file_path)

            if not file_path.exists():
                print("======= DOWNLOAD Zarr FROM GCS ======")
                gcs_path = self.GCS_BUCKET + "/" + resol + "/" + file_name
                print("DOWNLOAD: ", gcs_path)
                download_zarr(gcs_path, file_path)
        
        return dataset_path
    

    def load_dataset(self, dataset_path):
        print("데이터셋 불러오는 중...")
        self.datasets = []
        for path in dataset_path:
            ds, _ = xb.open_zarr(path)
            self.datasets.append(ds)


    def load_variable(self, data):
        data = data.to_numpy()

        data = torch.from_numpy(data)
        # data.shape = (time, width, height)
        # data.shape = (time, width * height)
        has_nan = torch.isnan(data).any()

        if has_nan:
            nan_indices = torch.isnan(data)
            data[nan_indices] = 0

        unnormalized = data.clone().detach()
        normalized = normalize_tensor(data)

        if len(data.shape) == 4:
            normalized = normalized.flatten(2)
            unnormalized = unnormalized.flatten(2)
        else:
            normalized = normalized.flatten(1)
            unnormalized = unnormalized.flatten(1)
        return normalized, unnormalized
    

    def load(self):
        result_dataset = []
        dims = []

        for dataset in self.datasets:
            result = self.load_data(dataset)
            result_dataset.append(result)
            dims.append(result.size(2))
        
        # var_dataset.shape = (time, var * level, h * w)
        result_dataset = torch.concat(result_dataset, dim=2)

        print("======= RESULT SHAPE =======")
        print("result_dataset.shape: ", result_dataset.shape)

        return result_dataset, dims

    
    def load_data(self, dataset:xr.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        start = time.time()
        result = {}

        print("==== LOAD DATASET ====\n", dataset)

        wind = (self.HAS_LEVEL_WIND_VAR + self.NONE_LEVEL_WIND_VAR)

        with ThreadPoolExecutor() as executor:
            futures = {}

            for val in (self.NONE_LEVEL_VARIABLE + self.HAS_LEVEL_VARIABLE):
                key = executor.submit(self.load_variable, dataset[val])
                futures[key] = val

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                val = futures[future]
                # shape => (level, time, h * w) or (time, h * w)
                normalized, unnormalized = future.result()
                if len(data.shape) == 3:
                    normalized = normalized.swapaxes(0, 1)
                    unnormalized = unnormalized.swapaxes(0, 1)
                result[val] = data
            
        wind_result = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            v1 = result[self.HAS_LEVEL_WIND_VAR[0]], result[self.HAS_LEVEL_WIND_VAR[1]]
            k1 = executor.submit(self.calculate_wind, v1[0], v1[1], self.device)
            futures[k1] = 1

            v2 = result[self.NONE_LEVEL_WIND_VAR[0]], result[self.NONE_LEVEL_WIND_VAR[1]]
            k2 = executor.submit(self.calculate_wind, v2[0], v2[1], self.device)
            futures[k2] = 0

            for future in tqdm(as_completed(futures), desc="Processing futures"):
                level = futures[future]
                # shape => (3, time, h * w) or (level * 3, time, h * w)
                data = future.result()
                if len(data.shape) == 4:
                    data = data.view(-1, data.size(2), data.size(3))
                wind_result[level] = data

            del result[self.HAS_LEVEL_WIND_VAR[0]]
            del result[self.HAS_LEVEL_WIND_VAR[1]]
            del result[self.NONE_LEVEL_WIND_VAR[0]]
            del result[self.NONE_LEVEL_WIND_VAR[1]]


        # dataset.shape => (var*level, time, h * w)
        dataset = []
        for val in (self.HAS_LEVEL_VARIABLE + self.NONE_LEVEL_VARIABLE):
            if val in (self.HAS_LEVEL_WIND_VAR + self.NONE_LEVEL_WIND_VAR):
                continue
            data = result[val]
            if len(data.shape) == 3:
                for i in range(data.size(0)):
                    dataset.append(data[i])
            else:
                dataset.append(data)

        dataset = torch.stack(dataset, dim=0)
        # dataset.shape => (time, var, h * w)
        dataset = torch.swapaxes(dataset, 0, 1)
        
        # wind.shape => (level, 3, time, h * w)
        wind_dataset = []
        wind_dataset.extend(wind_result[0])
        wind_dataset.extend(wind_result[1])

        # wind.shape => (level * 3, time, h * w)
        wind_dataset = torch.stack(wind_dataset, dim=0)
        # shape => (time, level * 3, h * w)
        wind_dataset = torch.swapaxes(wind_dataset, 0, 1)
        
        end = time.time()
        print(f"{end - start:.5f} sec")
        return torch.concat([dataset, wind_dataset], dim=1)


    def calculate_wind(self, u_wind, v_wind, device):
        res = preprocess_wind_data(u_wind, v_wind, device, self.normalize)
        torch.cuda.empty_cache()
        return res


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    device = torch.device(device)

    weather = WeatherDataset(0, device=device)
    weather.load()