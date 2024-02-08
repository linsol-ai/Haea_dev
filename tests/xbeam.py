import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray as xr
import xarray_beam as xb
import pandas as pd
import typing as t
import logging

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
    date_range = pd.date_range(start=start_date, end=end_date, inclusive='left')
    for date in date_range:
        yield date.year, date.month, date.day

class LoadTemporalDataForDateDoFn(beam.DoFn):
    """A Beam DoFn for loading temporal data for a specific date.

    This class is responsible for loading temporal data for a given date, including both
    single-level and pressure-level variables.
    Args:
        data_path (str): The path to the data source.
        start_date (str): The start date in ISO format (YYYY-MM-DD).
        pressure_levels_group (str): The group label for the set of pressure levels.
    Methods:
        process(args): Loads temporal data for a specific date and yields it with an xarray_beam key.
    Example:
        >>> data_path = "gs://your-bucket/data/"
        >>> start_date = "2023-09-01"
        >>> pressure_levels_group = "weatherbench_13"
        >>> loader = LoadTemporalDataForDateDoFn(data_path, start_date, pressure_levels_group)
        >>> for result in loader.process((2023, 9, 11)):
        ...     key, dataset = result
        ...     print(f"Loaded data for key: {key}")
        ...     print(dataset)
    """
    def __init__(self, data_path, start_date, pressure_levels_group):
        """Initialize the LoadTemporalDataForDateDoFn.
        Args:
            data_path (str): The path to the data source.
            start_date (str): The start date in ISO format (YYYY-MM-DD).
            pressure_levels_group (str): The group label for the set of pressure levels.
        """
        self.data_path = data_path
        self.start_date = start_date
        self.pressure_levels_group = pressure_levels_group

    def process(self, args):
        """Load temporal data for a day, with an xarray_beam key for it.
        Args:
            args (tuple): A tuple containing the year, month, and day.
        Yields:
            tuple: A tuple containing an xarray_beam key and the loaded dataset.
        """
        year, month, day = args
        logging.info("Loading zarr files for %d-%d-%d", year, month, day)

        try:
            single_level_vars = read_single_level_vars(
                year,
                month,
                day,
                variables=SINGLE_LEVEL_VARIABLES,
                root_path=self.data_path)
            multilevel_vars = read_multilevel_vars(
                year,
                month,
                day,
                variables=MULTILEVEL_VARIABLES,
                pressure_levels=get_pressure_levels_arg(self.pressure_levels_group),
                root_path=self.data_path)
        except BaseException as e:
            # Make sure we print the date as part of the error for easier debugging
            # if something goes wrong. Note "from e" will also raise the details of the
            # original exception.
            raise RuntimeError(f"Error loading {year}-{month}-{day}") from e

        # It is crucial to actually "load" as otherwise we get a pickle error.
        single_level_vars = single_level_vars.load()
        multilevel_vars = multilevel_vars.load()

        dataset = xr.merge([single_level_vars, multilevel_vars])
        dataset = align_coordinates(dataset)
        offsets = {"latitude": 0, "longitude": 0, "level": 0,
                   "time": offset_along_time_axis(self.start_date, year, month, day)}
        key = xb.Key(offsets, vars=set(dataset.data_vars.keys()))
        logging.info("Finished loading NetCDF files for %s-%s-%s", year, month, day)
        yield key, dataset
        dataset.close()'z'


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