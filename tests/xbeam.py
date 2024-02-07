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
    runner='DataflowRunner',
    project='genfit-7ba0d',
    temp_location=f'gs://{GCS_BUCKET}/temp',
    region='us-east1',
    requirements_file='/workspace/Haea/req.txt'
)

def get_chunk(dataset):
    import xarray_beam
    return xarray_beam.DatasetToChunks(dataset, chunks={'time': 10}, split_vars=False)

def save_chunk(dataset):
    import xarray_beam
    return xarray_beam.ChunksToZarr(dataset, OUTPUT_ZARR_PATH)

def preprocess_dataset(ds):
    ds_filtered = ds.sel(time=slice('2023-01-01', '2023-01-31'), lat=slice(30, 50), lon=slice(-130, -60))
    return ds_filtered

def run():
    
    with beam.Pipeline(options=options) as p:
        d1 = (
            p
            | 'OpenZarrDataset' >> beam.Create([xarray.open_zarr(INPUT_ZARR_PATH, chunks=None)])
            | 'ChunkingDataset' >> xarray_beam.DatasetToChunks(d1, chunks={'time': 10}, split_vars=False)
        )
        d2 = (
            d1
            | 'ChunkingDataset' >> xarray_beam.DatasetToChunks(d1, chunks={'time': 10}, split_vars=False)
            | 'PreprocessDataset' >> beam.Map(preprocess_dataset)
        )
        _ = (
            d2
            | 'WriteZarrToGCS' >> xarray_beam.ChunksToZarr(d2, OUTPUT_ZARR_PATH)
        )

        

if __name__ == '__main__':
    run()
