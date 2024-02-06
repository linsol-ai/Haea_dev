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
temporal_key = xbeam.Key('time')
spatial_keys = [xbeam.Key('latitude'), xbeam.Key('longitude')]

# Apache Beam 파이프라인 설정
pipeline_options = PipelineOptions()
with beam.Pipeline(options=pipeline_options) as p:
    # 데이터셋을 Beam PCollection으로 로드
    dataset = (
        p 
        | "Read Dataset" >> beam.Create([xr.open_dataset('your_dataset.nc')])
        | "Split into chunks" >> xbeam.SplitChunks(spatial_keys + [temporal_key])
    )

    # 필터링 작업 정의
    filtered_dataset = (
        dataset
        | "Filter by Time and Coordinates" >> beam.Filter(
            lambda chunk: temporal_key in chunk.indexes and 
                          chunk.indexes[temporal_key].min() >= pd.to_datetime('2021-01-01') and 
                          chunk.indexes[temporal_key].max() <= pd.to_datetime('2021-02-01') and
                          spatial_keys[0] in chunk.indexes and
                          chunk.indexes[spatial_keys[0]].min() >= lat_min and
                          chunk.indexes[spatial_keys[0]].max() <= lat_max and
                          spatial_keys[1] in chunk.indexes and
                          chunk.indexes[spatial_keys[1]].min() >= lon_min and
                          chunk.indexes[spatial_keys[1]].max() <= lon_max
        )
    )

    # 결과 처리 (예: 파일로 저장)
    # filtered_dataset | "Write Results" >> ...


if __name__ == '__main__':
  main()