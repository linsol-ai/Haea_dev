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
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


# pylint: disable=expression-not-assigned


def rekey_chunk_on_month_hour(
    key: xbeam.Key, dataset: xarray.Dataset
) -> Tuple[xbeam.Key, xarray.Dataset]:
    print(key, dataset)



def main(argv):
    source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
        
   

    with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
        (
            root
            | xbeam.DatasetToChunks(source_dataset, source_chunks)
            | xbeam.SplitChunks({'time': 10})
            | beam.MapTuple(rekey_chunk_on_month_hour)
        )


if __name__ == '__main__':
  app.run(main)