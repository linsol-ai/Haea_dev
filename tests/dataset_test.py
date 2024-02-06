import xarray_beam as xbeam
import xarray
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam
import pandas as pd
import numpy as np

# Xarray-Beam 파이프라인 정의
def run_pipeline(dataset:xarray.Dataset):

    pipeline = xbeam.DatasetToChunks(ds, chunks={'time': 20}, split_vars=False)
    pipeline |= beam.MapTuple(lambda k, v: print(k, type(v)))

    with beam.Pipeline(runner='DirectRunner',) as p:
        p | pipeline


# 파이프라인 실행
if __name__ == '__main__':
    ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-08-01')
    ds = ds.sel(time=slice(start_date, end_date))
    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131   
    lat_indices = np.where((ds.latitude >= lat_min) & (ds.latitude <= lat_max))[0]
    lon_indices = np.where((ds.longitude >= lon_min) & (ds.longitude <= lon_max))[0]
    ds = ds.isel(latitude=lat_indices, longitude=lon_indices)
    
    xbeam.DatasetToChunks(ds, chunks={'time': 20}, split_vars=False)