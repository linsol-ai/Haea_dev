import xarray_beam as xbeam
import xarray
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam
import pandas as pd

# Apache Beam 파이프라인 옵션 설정
pipeline_options = PipelineOptions(
    runner='DirectRunner',  # 로컬에서 실행하는 경우 'DirectRunner' 사용
    project='YOUR_PROJECT_ID',  # GCP 프로젝트 ID 설정
    region='YOUR_REGION',  # GCP 리전 설정
    temp_location='gs://YOUR_BUCKET/temp',  # 임시 파일 저장 위치 설정
)

# Xarray-Beam 파이프라인 정의
def run_pipeline():
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-08-01')
    ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
    ds = ds.sel(time=slice(start_date, end_date))
    with beam.Pipeline() as p:
        p | xbeam.DatasetToChunks(ds, chunks={'time': 1000}) | beam.Map(lambda chunk: print(chunk))


# 파이프라인 실행
if __name__ == '__main__':
    ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr', chunks='auto')
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-08-01')
    ds = ds.sel(time=slice(start_date, end_date))
    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131   
    lat_indices = np.where((ds.latitude >= lat_min) & (ds.latitude <= lat_max))[0]
    lon_indices = np.where((ds.longitude >= lon_min) & (ds.longitude <= lon_max))[0]
    ds = ds.isel(latitude=lat_indices, longitude=lon_indices)

    run_pipeline()