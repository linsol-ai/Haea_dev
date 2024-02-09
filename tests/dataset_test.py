import xarray
import pandas as pd
import numpy as np
import time
import xarray_beam



# 파이프라인 실행
if __name__ == '__main__':

    start = time.time()
    ds = xarray.open_zarr('gs://era5_preprocess/test/test.zarr', 
                          consolidated=True, 
                          chunks=None,
                          )
    print(ds)
