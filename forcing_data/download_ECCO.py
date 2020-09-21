import pandas as pd
import subprocess
import os

YR_START = '2015'  # inclusive
YR_END = '2016'  # not inclusive
VARNAMES = ['NVEL', 'EVEL', 'WVELMASS']
OUT_DIR = f'ECCO/'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

for date in pd.date_range(YR_START, YR_END, freq='D', closed='left'):
    for VARNAME in VARNAMES:
        filename = f'{VARNAME}_{date.strftime("%Y_%m_%d")}.nc'
        if os.path.exists(OUT_DIR + filename):
            continue
        url = f'https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_90/ECCOv4/Release4/nctiles_daily/' \
              f'{VARNAME}/{date.year}/{date.dayofyear:03}/{filename}'

        subprocess.run(['wget', '--retry-connrefused', '--waitretry=1', '--read-timeout=20', '--timeout=15', '-t 0',
                        '-q', '--show-progress', '-O', OUT_DIR + filename, url])


# occasionally you'll get incompletely downloaded files.  Run this command to see if there are any files less than 20 MB
# find . -name "*.nc" -type 'f' -size -20M
# then run this command to remove them
# find . -name "*.nc" -type 'f' -size -20M -delete
# once removed, run this script again and it will re-download the deleted files
