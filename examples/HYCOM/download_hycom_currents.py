import wget
import pandas as pd
import sys
import os
import glob
from tqdm import tqdm

from os import path


for time in tqdm(pd.date_range(start='2015-01-01', end='2016-01-01', closed='left', freq='1D')):
    url = 'http://ncss.hycom.org/thredds/ncss/grid/GLBu0.08/expt_91.1/uv3z?var=water_u&var=water_v' \
         f'&north=80&west=0&east=359.92&south=-80&horizStride=1&time={time.strftime("%Y-%m-%dT%H")}' \
          '%3A00%3A00Z&vertCoord=0&accept=netcdf'
    filename = path.join(path.dirname(__file__), f'currents/uv_surf_HYCOM_{time.strftime("%Y-%m-%dT%H")}.nc')

    if not path.exists(filename):
        sys.stdout = open(os.devnull, "w")  # this suppresses annoying wget messages
        wget.download(url, filename)
        sys.stdout = sys.__stdout__


# clean up uncompleted downloads
for file in glob.glob('data/HYCOM/*.tmp'):
    os.remove(file)
