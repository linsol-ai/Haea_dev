
from typing import Tuple

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


# pylint: disable=expression-not-assigned


def rekey_chunk_on_month_hour(
    key: xbeam.Key, dataset: xarray.Dataset
) -> Tuple[xbeam.Key, xarray.Dataset]:
  """Replace the 'time' dimension with 'month'/'hour'."""
  month = dataset.time.dt.month.item()
  hour = dataset.time.dt.hour.item()
  new_key = key.with_offsets(time=None, month=month - 1, hour=hour)
  new_dataset = dataset.squeeze('time', drop=True).expand_dims(
      month=[month], hour=[hour]
  )
  return new_key, new_dataset


def main(argv):
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)

  start_date = pd.to_datetime('2021-01-01')
end_date = pd.to_datetime('2021-08-01')
ds = ds.sel(time=slice(start_date, end_date))
  
  template = (
      xbeam.make_template(source_dataset)
      .isel(time=0, drop=True)
      .expand_dims(month=np.arange(1, max_month + 1), hour=np.arange(24))
  )
  output_chunks = {'time': 1, 'month': 1}

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | xbeam.SplitChunks({'time': 1})
        | beam.MapTuple(rekey_chunk_on_month_hour)
        | xbeam.Mean.PerKey()
        | xbeam.ChunksToZarr(OUTPUT_PATH.value, template, output_chunks)
    )


if __name__ == '__main__':
  app.run(main)