#%%
import xarray as xr
import numpy as np
import os
import re
# import pandas as pd
from dask.distributed import Client
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import DecomposeResult

#%% # Explicit definition of local dask cluster
client = Client(processes=False, n_workers = 2)
client

# %% # load data
datapath = '../../dat/TRAINING/ERA5/'
files = [f for f in os.listdir(datapath) if re.match(r't2m_era5_[0-9]+.*\.nc', f)]
dat = xr.open_mfdataset([os.path.join(datapath, f) for f in files])
# chunking is problematic as time dimension is chunked (specifying chunks in open_mfdataset cannot create chunk sizes bigger than the dimensions in each file)
# rechunking is not recommended for data loaded with open_mfdataset and leads to performance issues
dat = dat.chunk(time = -1, longitude = 1, latitude = 1)

# %% TODO: MSTL works on single pixel
testdat = dat.t2m.isel(latitude = 100, longitude = 100).load()
# %% TODO: use wider windows to not let the trend component include seasonality
test = MSTL(testdat, periods=(8, 365)).fit()
test.plot()

# %% # timeseries decomposition
def decompose(xds):
    def MSTLarray(ts): 
        # TODO: does not work yet
        out = MSTL(ts, periods=(8, 2920)).fit() # 3 hourly data
        return out.trend, out.seasonal[:, 0], out.seasonal[:, 1], out.resid
        # return ts, ts * 2, ts * 3, ts * 4 # comment out above lines and use this dummy output for testing the ufunc declaration
    trend, seas, seas2, resid =  xr.apply_ufunc(
        MSTLarray,
        xds.t2m,
        dask="parallelized",
        input_core_dims=[["time"]],
        output_core_dims=[["time"],["time"], ["time"], ["time"]],
        vectorize=True,
    )
    out = trend.to_dataset(name = "trend")
    out["seasonality_diurnal"] = seas
    out["seasonality_annual"] = seas2
    out["residual"] = resid
    return out

out =  decompose(dat)
# %% save output to disk
# TODO: fist solve todos, then uncomment
# TODO: again save to multiple files?!
# out.to_netcdf("../../dat/DECOMPOSED/ERA5/ERA5_decomposed.nc")


# TODO:
# - chunking: no chunking for time dimension?!
# - MSTL: period does not respect leap years
# - MSTL: performance - to pandas and back not optimal