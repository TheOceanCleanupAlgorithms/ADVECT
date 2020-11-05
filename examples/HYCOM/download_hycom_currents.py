"""this downloads 2015 daily surface currents from HYCOM u0.08 expt 91.1"""

import wget
import pandas as pd
import sys
import os
import glob
import xarray as xr
import datetime
from tqdm import tqdm

missing_days = [datetime.date.fromisoformat(datestr)
                for datestr in ('2015-01-02', '2015-03-15', '2015-03-25', '2015-09-19')]

out_dir = os.path.join(os.path.dirname(__file__), 'currents')
for time in tqdm(pd.date_range(start='2015-01-01T00', end='2016-01-01T00', closed='left', freq='1D')):
    url = 'http://ncss.hycom.org/thredds/ncss/grid/GLBu0.08/expt_91.1/uv3z?var=water_u&var=water_v' \
         f'&north=80&west=0&east=359.92&south=-80&horizStride=1&time={time.strftime("%Y-%m-%dT%H")}' \
          '%3A00%3A00Z&vertCoord=0&accept=netcdf'
    filename = os.path.join(out_dir, f'uv_surf_HYCOM_{time.strftime("%Y-%m-%dT%H")}.nc')

    if not os.path.exists(filename):
        sys.stdout = open(os.devnull, "w")  # this suppresses annoying wget messages
        wget.download(url, filename)
        sys.stdout = sys.__stdout__

        # this hycom experiment has missing days in 2015.  This script still fetches files for these days,
        # but their 'time' value is set as the day before.
        # Let's 'fix' this, as ADVECTOR requires a uniformly spaced time dimension.  This is 'sort of' interpolating.
        if time.date() in missing_days:
            ds = xr.open_dataset(filename)
            ds['time'] = ds.time + pd.Timedelta('1 day')
            ds.to_netcdf(filename)

# clean up uncompleted downloads
for file in glob.glob('data/HYCOM/*.tmp'):
    os.remove(file)
