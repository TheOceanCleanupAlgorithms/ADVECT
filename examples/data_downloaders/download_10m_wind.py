"""
simple script to download daily-mean 10m wind from the ncep ncar doe ii reanalysis
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr


def download_and_interpolate_ncep_ncar_wind(out_dir: Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        urls = [
            "ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis2.dailyavgs/gaussian_grid/uwnd.10m.gauss.2015.nc",
            "ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis2.dailyavgs/gaussian_grid/vwnd.10m.gauss.2015.nc",
        ]
        filenames = [url.split("/")[-1] for url in urls]

        for filename, url in zip(filenames, urls):
            out_path = out_dir / filename
            print(out_path)
            if out_path.exists():
                print(f"skipping {out_path}, already exists")
                continue
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
                    str(Path(temp_dir) / filename),
                    url,
                ]
            )

            # interpolate to regular grids
            print(f"Interpolating latitude to evenly spaced grid for file {filename}")
            ds = xr.open_dataset(Path(temp_dir) / filename)
            ds = ds.sortby("lat", ascending=True)
            ds = ds.interp(lat=np.linspace(ds.lat.min(), ds.lat.max(), len(ds.lat)))
            ds.to_netcdf(out_path)
