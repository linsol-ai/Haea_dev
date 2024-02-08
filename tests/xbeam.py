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
OUTPUT_PATH = 'gs://dataflow_preprocess/preprocessed'
RUNNER = 'DirectRunner'

# pylint: disable=expression-not-assigned


def rekey_chunk_on_month_hour(
    key: xbeam.Key, dataset: xarray.Dataset
) -> Tuple[xbeam.Key, xarray.Dataset]:
  """Replace the 'time' dimension with 'month'/'hour'."""
  start_date = pd.to_datetime('2021-01-01')
  end_date = pd.to_datetime('2021-08-01')
  dataset = dataset.sel(time=slice('2021-01-01', '2022-01-31'), lat=slice(30, 50), lon=slice(-130, -60))

  month = dataset.time.dt.month.item()
  hour = dataset.time.dt.hour.item()
  new_key = key.with_offsets(time=None, month=month - 1, hour=hour)
  new_dataset = dataset.squeeze('time', drop=True).expand_dims(
      month=[month], hour=[hour]
  )
  return new_key, new_dataset


def main(argv):
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH)

  # This lazy "template" allows us to setup the Zarr outputs before running the
  # pipeline. We don't really need to supply a template here because the outputs
  # are small (the template argument in ChunksToZarr is optional), but it makes
  # the pipeline slightly more efficient.
  max_month = source_dataset.time.dt.month.max().item()  # normally 12
  template = (
      xbeam.make_template(source_dataset)
      .isel(time=0, drop=True)
      .expand_dims(month=np.arange(1, max_month + 1), hour=np.arange(24))
  )

  with beam.Pipeline(runner=RUNNER, argv=argv) as root:
    (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | xbeam.SplitChunks({'time': 1})
        | beam.MapTuple(rekey_chunk_on_month_hour)
        | xbeam.Mean.PerKey()
        | xbeam.ChunksToZarr(OUTPUT_PATH, template, None)
    )


if __name__ == '__main__':
  app.run(main)