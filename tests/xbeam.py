import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
x
import xarray as xr
import fsspec

# GCS 경로 설정
GCS_BUCKET = 'YOUR_BUCKET_NAME'
INPUT_ZARR_PATH = 'path/to/your/input/data.zarr'
OUTPUT_ZARR_PATH = 'path/to/your/output/data.zarr'

# 파이프라인 옵션 설정
options = PipelineOptions(
    runner='DataflowRunner',
    project='YOUR_PROJECT_ID',
    temp_location=f'gs://{GCS_BUCKET}/temp',
    region='YOUR_REGION'
)

def preprocess_dataset(element):
    # Xarray 데이터셋 전처리 로직
    # 예: 시간 및 공간 범위 필터링, 평균 계산 등
    ds = element[1]
    ds_filtered = ds.sel(time=slice('2023-01-01', '2023-01-31'), lat=slice(30, 50), lon=slice(-130, -60))
    return ds_filtered

def run():
    with beam.Pipeline(options=options) as p:
        _ = (
            p
            | 'CreateDatasetPattern' >> beam.Create([f'gs://{GCS_BUCKET}/{INPUT_ZARR_PATH}'])
            | 'OpenZarrDataset' >> xbeam.OpenZarr()
            | 'PreprocessDataset' >> beam.Map(preprocess_dataset)
            | 'WriteZarrToGCS' >> xbeam.WriteZarr(f'gs://{GCS_BUCKET}/{OUTPUT_ZARR_PATH}', 
                                                  template_ds=None)  # template_ds 설정 필요에 따라 조정
        )

if __name__ == '__main__':
    run()
