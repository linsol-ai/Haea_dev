import numpy as np
from tqdm import tqdm

import numpy as np
import xarray_beam as xbeam
import xarray
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
import apache_beam as beam

ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr', chunks={'time': 1000})
start_date = pd.to_datetime('2021-01-01')
end_date = pd.to_datetime('2021-08-01')
ds = ds.sel(time=slice(start_date, end_date))

with beam.Pipeline() as p:
    p | xbeam.DatasetToChunks(ds) | beam.MapTuple(lambda k, v: print(k, type(v)))