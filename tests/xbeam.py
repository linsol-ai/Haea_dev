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

def daily_date_iterator(start_date: str, end_date: str
                        ) -> t.Iterable[t.Tuple[int, int, int]]:
    """Iterate through all (year, month, day) tuples between start_date and
    end_date (inclusive).

    Args:
        start_date (str): The start date in ISO format (YYYY-MM-DD).
        end_date (str): The end date in ISO format (YYYY-MM-DD).

    Yields:
        tuple: A tuple containing the year, month, and day for each date in the range.

    Example:
        >>> for year, month, day in daily_date_iterator('2023-09-01', '2023-09-05'):
        ...     print(f"Year: {year}, Month: {month}, Day: {day}")
        Year: 2023, Month: 9, Day: 1
        Year: 2023, Month: 9, Day: 2
        Year: 2023, Month: 9, Day: 3
        Year: 2023, Month: 9, Day: 4
        Year: 2023, Month: 9, Day: 5
    """
    date_range = pd.date_range(start=start_date, end=end_date, inclusive='left')
    for date in date_range:
        yield date.year, date.month, date.day


def get_chunk(dataset):
    import xarray_beam
    return xarray_beam.DatasetToChunks(dataset, chunks={'time': 10}, split_vars=False)

def save_chunk(dataset):
    import xarray_beam
    return xarray_beam.ChunksToZarr(dataset, OUTPUT_ZARR_PATH)

def preprocess_dataset( key: xarray_beam.Key, dataset: xarray.Dataset):
    ds = dataset
    ds_filtered = ds.sel(time=slice('2023-01-01', '2023-01-31'), latitude=slice(32.2, 39.0), longitude=slice(124.2, 131))
    return key, ds_filtered.compute()

def run():
    source_dataset, source_chunks = xarray_beam.open_zarr(INPUT_ZARR_PATH)
    with beam.Pipeline(options=options) as p:
        _ = (
            p
            | 'ChunkingDataset' >> xarray_beam.DatasetToChunks(source_dataset, chunks={'time': 10}, split_vars=False)
            | 'PreprocessDataset' >> beam.MapTuple(preprocess_dataset)
            | 'WriteZarrToGCS' >> xarray_beam.ChunksToZarr('/workspace/Haea/tests/1440x721.zarr')
        )

if __name__ == '__main__':
    run()