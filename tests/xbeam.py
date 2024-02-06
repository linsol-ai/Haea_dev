# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Calculate climatology for the Pangeo ERA5 surface dataset."""
from typing import Tuple

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam
import pandas as pd


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('example-data-v3.zarr', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


# pylint: disable=expression-not-assigned


def rekey_chunk_on_month_hour(
    key: xbeam.Key, dataset: xarray.Dataset):
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-02-01')
    arr = dataset.sel(time=slice(start_date, end_date))
    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131

    # isel 함수 대신 sel 함수를 사용하여 경위도 범위를 필터링
    arr = arr.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)) 
    



def main():
    source_dataset, source_chunks = xbeam.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr')

    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-02-01')
    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131

    template = (
      xbeam.make_template(source_dataset)
      .sel(time=slice(start_date, end_date))
      .sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    )

    with beam.Pipeline(runner='DirectRunner') as root:
        (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | beam.MapTuple(rekey_chunk_on_month_hour)
        | xbeam.SplitChunks({'time': 10})
        ) 

    print(template)


if __name__ == '__main__':
  main()