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

INPUT_PATH = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr'
OUTPUT_PATH = 'gs://dataflow_preprocess/preprocessed/test.zarr'
RUNNER = 'DirectRunner'

# pylint: disable=expression-not-assigned


def rekey_chunk_on_month_hour(
    key: xbeam.Key, dataset: xarray.Dataset
) -> Tuple[xbeam.Key, xarray.Dataset]:
  """Replace the 'time' dimension with 'month'/'hour'."""
  new_dataset = dataset.sel(time=slice('2021-01-01', '2022-01-01'), latitude=slice(32.2, 39.0), longitude=slice(124.2, 131))
  return key, new_dataset


def main(argv):
  variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH)
  source_dataset = source_dataset[variable]
  template = (
      xbeam.make_template(source_dataset)
      .isel(time=0, drop=True)
      .expand_dims(month=np.arange(1, max_month + 1), hour=np.arange(24))
  )
  output_chunks = {'hour': 1, 'month': 1}

  with beam.Pipeline(runner=RUNNER, argv=argv) as root:
    (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | xbeam.SplitChunks({'time': 1})
        | beam.MapTuple(rekey_chunk_on_month_hour)
        | xbeam.Mean.PerKey()
        | xbeam.ChunksToZarr(OUTPUT_PATH, template, output_chunks)
    )


if __name__ == '__main__':
  app.run(main)