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


def main():

    # 데이터셋을 분할하기 위한 키 설정

    lat_min, lat_max = 32.2, 39.0
    lon_min, lon_max = 124.2, 131
    source_dataset, source_chunks = xbeam.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr')

    print(source_chunks)

    # Apache Beam 파이프라인 설정
    pipeline_options = PipelineOptions()
    with beam.Pipeline(options=pipeline_options) as p:
        # 데이터셋을 Beam PCollection으로 로드
        dataset = (
            p 
            | "Read Dataset" >> xbeam.DatasetToChunks(source_dataset, source_chunks)
            | "Split into chunks" >> xbeam.SplitChunks({'time':10})
        )

    
        print(dataset)


if __name__ == '__main__':
  main()