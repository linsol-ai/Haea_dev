import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray
import xarray_beam

# GCS 경로 설정
GCS_BUCKET = 'dataflow_preprocess'
INPUT_ZARR_PATH = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
OUTPUT_ZARR_PATH = f'gs://{GCS_BUCKET}/result/1440x721.zarr'

# 파이프라인 옵션 설정
options = PipelineOptions(
    runner='DirectRunner',
)

def get_chunk(dataset):
    import xarray_beam
    return xarray_beam.DatasetToChunks(dataset, chunks={'time': 10}, split_vars=False)

def save_chunk(dataset):
    import xarray_beam
    return xarray_beam.ChunksToZarr(dataset, OUTPUT_ZARR_PATH)

def preprocess_dataset(element):
    ds = element[1] 
    ds_filtered = ds.sel(time=slice('2023-01-01', '2023-01-31'), lat=slice(30, 50), lon=slice(-130, -60))
    return ds_filtered

def run():
    dataset = xarray.open_zarr(INPUT_ZARR_PATH, chunks=None)
    with beam.Pipeline(options=options) as p:
        _ = (
            p
            | 'ChunkingDataset' >> xarray_beam.DatasetToChunks(dataset, chunks={'time': 10}, split_vars=False)
            | 'PreprocessDataset' >> beam.Map(preprocess_dataset)
            | 'WriteZarrToGCS' >> xarray_beam.ChunksToZarr(OUTPUT_ZARR_PATH)
        )

        

if __name__ == '__main__':
    run()
