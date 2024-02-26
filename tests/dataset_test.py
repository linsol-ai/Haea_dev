import xarray
import pandas as pd
import numpy as np
import time
import xarray_beam


HAS_LEVEL_VARIABLE = [ 'geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

NONE_LEVEL_VARIABLE = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'sea_surface_temperature', 'total_cloud_cover', 'total_precipitation_24hr', 'total_precipitation_6hr']

# 파이프라인 실행
if __name__ == '__main__':

    start = time.time() 
    ds = xarray.open_zarr('gs://era5_preprocess/1440x720/2016-12-31_2021-12-31.zarr')
    print(ds.nbytes / (10**9))
    print(ds)

    HAS_LEVEL_VARIABLE = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

    # 새 차원을 추가하고 데이터 변수를 결합
    data_arrays = [ds[var].expand_dims('variable').assign_coords(variable=[var]) for var in HAS_LEVEL_VARIABLE]
    combined_ds = xarray.concat(data_arrays, dim='variable')
    # 결과 확인
    stacked_ds = combined_ds.stack(variable_level=('variable', 'level'))
    stacked_ds = stacked_ds.stack(hidden_dim=('latitude', 'longitude'))
    print(stacked_ds.values.shape)
    print(ds.values)
    end = time.time()
    print(f"{end - start:.5f} sec")
