
from typing import Tuple
from apache_beam.options.pipeline_options import PipelineOptions
from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam
import pandas as pd


RUNNER = 'DirectRunner'
HAS_LEVEL_VARIABLE = [
  'geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

NONE_LEVEL_VARIABLE = [
  'toa_incident_solar_radiation', 'land_sea_mask', '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'sea_surface_temperature', 'total_cloud_cover', 'total_precipitation'
]

VARIABLE = HAS_LEVEL_VARIABLE + NONE_LEVEL_VARIABLE

LAT = [(32, 39.3), (20, 50), (0, 70)]
LON = [(124, 131.4), (115, 145), (90, 180)]

INPUT_PATHS = [
  'gcs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
  'gcs://gcp-public-data-arco-era5/ar/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr',
  'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'
]
FOLDER_NAME = ['1440x720', '240x120', '60x30']


# pylint: disable=expression-not-assigned
FLAGS = flags.FLAGS
START_YEAR = flags.DEFINE_string('start', None, help='start_year')
END_YEAR = flags.DEFINE_string('end', None, help='end_year')
TYPE = flags.DEFINE_integer('type', None, help='type 0 : 1440x720, type 1 : 240x120, type 2 : 60x30')

flags.mark_flag_as_required("start")
flags.mark_flag_as_required("end")
flags.mark_flag_as_required("type")


def rekey_chunk_on_month_hour(
    key: xbeam.Key, dataset: xarray.Dataset, type=0, lat_indices=None, lon_indices=None) -> Tuple[xbeam.Key, xarray.Dataset]:
    if type == 0:
        dataset = dataset.isel(latitude=lat_indices, longitude=lon_indices).sortby('latitude', ascending=True)
    elif type == 1:
        dataset = dataset.isel(latitude=lat_indices, longitude=lon_indices).transpose('time', 'level', 'latitude', 'longitude')

    new_key = key.with_offsets(time=128)
    return new_key, dataset


def main(argv):
  START_DATE = f'{FLAGS.start}-12-31'
  END_DATE = f'{FLAGS.end}-12-31'
  print('Preprocess Data: ', START_DATE, 'to', END_DATE)
  OUTPUT_PATH = f'gs://era5_climate/{FOLDER_NAME[FLAGS.type]}/{START_DATE}_{END_DATE}.zarr'

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATHS[FLAGS.type])
  source_dataset = source_dataset[VARIABLE]

  start_date = pd.to_datetime(START_DATE)
  end_date = pd.to_datetime(END_DATE)

  lat_min, lat_max = LAT[FLAGS.type]
  lon_min, lon_max = LON[FLAGS.type]

  # 해당 범위에 속하는 위도와 경도의 인덱스 찾기
  lat_indices = np.where((source_dataset.latitude >= lat_min) & (source_dataset.latitude <= lat_max))[0]
  lon_indices = np.where((source_dataset.longitude >= lon_min) & (source_dataset.longitude <= lon_max))[0]

  source_dataset = source_dataset.sel(time=slice(start_date, end_date))

  output_chunks = source_chunks.copy()
  output_chunks['time'] = 128

  template = None

  if FLAGS.type == 0:
      dataset = dataset.isel(latitude=lat_indices, longitude=lon_indices).sortby('latitude', ascending=True)
  elif FLAGS.type == 1:
      dataset = dataset.isel(latitude=lat_indices, longitude=lon_indices).transpose('time', 'level', 'latitude', 'longitude')

  template = (
      xbeam.make_template(source_dataset)
  )

  pipeline_options = PipelineOptions(
        runner='DataflowRunner',
        project='climate-414222',
        temp_location='gs://era5_climate/temp',
        requirements_file='/workspace/Haea_dev/req.txt',
        region='us-central1',
        machine_type='c3d-standard-90',
  )

  with beam.Pipeline(options=pipeline_options) as root :
    (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | xbeam.SplitChunks({'time': 1})
        | beam.MapTuple(rekey_chunk_on_month_hour, FLAGS.type, lat_indices, lon_indices)
        | xbeam.ConsolidateChunks({'time':128})
        | xbeam.ChunksToZarr(OUTPUT_PATH, template, { 'time':128 })
    )


if __name__ == '__main__':
  app.run(main)