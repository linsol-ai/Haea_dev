
from typing import Tuple
from apache_beam.options.pipeline_options import PipelineOptions
from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam
import pandas as pd



INPUT_PATH = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
RUNNER = 'DirectRunner'
HAS_LEVEL_VARIABLE = [
  'geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

NONE_LEVEL_VARIABLE = [
  '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'sea_surface_temperature', 'total_cloud_cover', 'total_precipitation_24hr', 'total_precipitation_6hr'
]
VARIABLE = HAS_LEVEL_VARIABLE + NONE_LEVEL_VARIABLE

# pylint: disable=expression-not-assigned

START_YEAR = flags.DEFINE_string('start', None, help='start_year')
END_YEAR = flags.DEFINE_string('end', None, help='end_year')
NUM_WORKER = flags.DEFINE_integer('num_worker', 12, help='num_worker')

flags.mark_flag_as_required("start")
flags.mark_flag_as_required("end")


def rekey_chunk_on_month_hour(
    key: xbeam.Key, dataset: xarray.Dataset, lat_indices=None, lon_indices=None
) -> Tuple[xbeam.Key, xarray.Dataset]:
  """Replace the 'time' dimension with 'month'/'hour'."""
  new_dataset = dataset.isel(latitude=lat_indices, longitude=lon_indices)
  return key, new_dataset


def main(argv):
  START_DATE = f'{START_YEAR}-01-01'
  END_DATE = f'{END_YEAR}-01-01'
  print(ST)
  OUTPUT_PATH = f'gs://era5_preprocess/1440x721/{START_DATE}_{END_DATE}.zarr'

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH)
  source_dataset = source_dataset[VARIABLE]
  start_date = pd.to_datetime(START_DATE)
  end_date = pd.to_datetime(END_DATE)
  source_dataset = source_dataset.sel(time=slice(start_date, end_date))

  lat_min, lat_max = 32.2, 39.0
  lon_min, lon_max = 124.2, 131

  # 해당 범위에 속하는 위도와 경도의 인덱스 찾기
  lat_indices = np.where((source_dataset.latitude >= lat_min) & (source_dataset.latitude <= lat_max))[0]
  lon_indices = np.where((source_dataset.longitude >= lon_min) & (source_dataset.longitude <= lon_max))[0]
  
  template = (
      xbeam.make_template(source_dataset)
      .isel(latitude=lat_indices, longitude=lon_indices)
  )

  output_chunks = {'time': 1}

  pipeline_options = PipelineOptions(
        runner='DataflowRunner',
        project='genfit-7ba0d',
        temp_location='gs://era5_preprocess/temp',
        region='us-central1',
        requirements_file='/workspace/Haea/req.txt',
        num_workers=NUM_WORKER
  )


  with beam.Pipeline(options=pipeline_options) as root:
    (
        root
        | xbeam.DatasetToChunks(source_dataset, {'time':1})
        | beam.MapTuple(rekey_chunk_on_month_hour, lat_indices=lat_indices, lon_indices=lon_indices)
        | xbeam.ChunksToZarr(OUTPUT_PATH, template, output_chunks)
    )


if __name__ == '__main__':
  app.run(main)