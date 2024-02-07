import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray as xr

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
    ds = element[1]
    ds_filtered = ds.sel(time=slice('2023-01-01', '2023-01-31'), lat=slice(30, 50), lon=slice(-130, -60))
    return ds_filtered

def run():
    with beam.Pipeline(options=options) as p:
        _ = (
            p
            | 'CreateDatasetPattern' >> beam.Create([f'gs://{GCS_BUCKET}/{INPUT_ZARR_PATH}'])
            | 'OpenZarrDataset' >> xr.open_zarr(INPUT_ZARR_PATH)
            | 'PreprocessDataset' >> beam.Map(preprocess_dataset)
            | 'WriteZarrToGCS' >> xarray_beam.ChunksToZarr(OUTPUT_ZARR_PATH)
        )

if __name__ == '__main__':
    run()
