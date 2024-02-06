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
    ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr', chunks={'time': 10, 'latitude': 100, 'longitude': 100, 'level': 1})

    print(f'size: {ds.nbytes / (1024 ** 4)} TiB')
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-12-01')
    variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

    arr = ds.sel(time=slice(start_date, end_date))
    lat_min, lat_max = 24.5, 44.0
    lon_min, lon_max = 120, 139.5

    # isel 함수 대신 sel 함수를 사용하여 경위도 범위를 필터링
    arr = arr.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    arr = arr['geopotential']
    print(arr)
    # 새 차원을 추가하고 데이터 변수를 결합

    data_arrays = [arr[var].expand_dims('variable').assign_coords(variable=[var]) for var in variable]
    combined_ds = xarray.concat(data_arrays, dim='variable')
    # 결과 확인
    print(combined_ds)
    stacked_ds = combined_ds.stack(variable_level=('variable', 'level'))
    stacked_ds = stacked_ds.stack(hidden_dim=('latitude', 'longitude'))
    arr = stacked_ds.to_numpy()

    end = time.time()
    print(f"{end - start:.5f} sec")
    print(arr)

