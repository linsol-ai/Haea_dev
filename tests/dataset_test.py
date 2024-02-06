import numpy as np
from tqdm import tqdm

import numpy as np
import xarray_beam as xbeam
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch

ds, chunks = xbeam.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
