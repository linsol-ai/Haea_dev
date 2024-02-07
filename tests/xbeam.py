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
import xarray_beam as xbeam
import xarray as xr
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']


def proprocess_data(
    key: xbeam.Key, dataset: xarray.Dataset
) -> Tuple[xbeam.Key, xarray.Dataset]:

    dataset = dataset[variable]
    
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2021-02-01')
   
    dataset = dataset.sel(time=slice(start_date, end_date))
    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131

    # isel 함수 대신 sel 함수를 사용하여 경위도 범위를 필터링
    dataset = dataset.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    return key, dataset



def main():

    source_dataset, source_chunks = xbeam.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr')

    print(source_chunks)

    # Apache Beam 파이프라인 설정
    options = PipelineOptions(
        runner='DataflowRunner',
        project='genfit-7ba0d',
        job_name='unique-job-name',
        temp_location='gs://dataflow_preprocess/test',
        region='us-east1'
    )

    with beam.Pipeline(options=options) as p:
        # 데이터셋을 Beam PCollection으로 로드
        (
            p 
            | "Read Dataset" >> xbeam.DatasetToChunks(source_dataset, {'time': 10}, split_vars=False,)
            | "Preprocess Dataset" >> beam.MapTuple(proprocess_data)
            | "Save Dataset" >> xbeam.ChunksToZarr('6h-1440x721.zarr')
        )



if __name__ == '__main__':
  main()