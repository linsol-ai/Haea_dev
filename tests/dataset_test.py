import xarray_beam as xbeam
import xarray
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam
import pandas as pd
import numpy as np
import time


# 파이프라인 실행
if __name__ == '__main__':
    start = time.time()
    ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr', chunks={'latitude': 10, 'longitude': 10}

)
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-12-01')
    variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
    arr = ds['geopotential']

    arr = arr.sel(time=slice(start_date, end_date))
    lat_min, lat_max = 24.5, 44.0
    lon_min, lon_max = 120, 139.5

    # isel 함수 대신 sel 함수를 사용하여 경위도 범위를 필터링
    arr = arr.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    print(arr)
    # 새 차원을 추가하고 데이터 변수를 결합

    arr = arr.to_numpy()

    end = time.time()
    print(f"{end - start:.5f} sec")
    print(arr.shape)

