import xarray
import pandas as pd
import numpy as np
import time
import xarray_beam



# 파이프라인 실행
if __name__ == '__main__':

    start = time.time()
    ds = xarray.open_zarr('gs://era5_preprocess/1440x720/2018-01-01_2023-01-01.zarr', chunks={'time':10})
    
    print(ds)
