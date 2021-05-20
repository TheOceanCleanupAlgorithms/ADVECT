"""
simple script to download daily-mean 10m wind from the ncep ncar doe ii reanalysis
"""

import subprocess
import xarray as xr
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent
U = {
    "path": str(OUT_DIR / "uwnd.10m.gauss.2015.nc"),
    "url": "ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis2.dailyavgs/gaussian_grid/uwnd.10m.gauss.2015.nc",
}
V = {
    "path": str(OUT_DIR / "vwnd.10m.gauss.2015.nc"),
    "url": "ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis2.dailyavgs/gaussian_grid/vwnd.10m.gauss.2015.nc",
}

for wind in [U, V]:
    subprocess.run(
        [
            "wget",
            "--retry-connrefused",
            "--waitretry=1",
            "--read-timeout=20",
            "--timeout=15",
            "-t 0",
            "-q",
            "--show-progress",
            "-O",
            str(OUT_DIR / wind["path"]),
            wind["url"],
        ]
    )

# interpolate to regular grids
for wind in [U, V]:
    print(f'Interpolating latitude to evenly spaced grid for file {U["path"]}')
    ds = xr.open_dataset(wind["path"]).load()
    ds.close()
    ds = ds.sortby("lat", ascending=True)
    ds = ds.interp(lat=np.linspace(ds.lat.min(), ds.lat.max(), len(ds.lat)))
    ds.to_netcdf(wind["path"])
