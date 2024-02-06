import xarray_beam as xbeam
import xarray
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam
import pandas as pd
import numpy as np


# 파이프라인 실행
if __name__ == '__main__':
    ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr', chunks={'time': 50})
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-08-01')
    ds = ds.sel(time=slice(start_date, end_date))
    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131
    lat_indices = np.where((ds.latitude >= lat_min) & (ds.latitude <= lat_max))[0]
    lon_indices = np.where((ds.longitude >= lon_min) & (ds.longitude <= lon_max))[0]
    ds = ds.isel(latitude=lat_indices, longitude=lon_indices)

    variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
    arr = ds[variable]
    # 새 차원을 추가하고 데이터 변수를 결합
    data_arrays = [arr[var].expand_dims('variable').assign_coords(variable=[var]) for var in variable]
    combined_ds = xarray.concat(data_arrays, dim='variable')
    # 결과 확인
    stacked_ds = combined_ds.stack(variable_level=('variable', 'level'))
    stacked_ds = stacked_ds.stack(hidden_dim=('latitude', 'longitude'))
    arr = stacked_ds.compute()
    print(arr)

