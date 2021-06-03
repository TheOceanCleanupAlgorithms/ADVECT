import glob
import os
from pathlib import Path

import xarray as xr


def process_density(
    density_dir: Path,
    out_path: Path,
):
    """combine all the density anomaly files into one absolute density file, with clean coordinates"""
    rhoConst = 1029  # source: https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/nctiles_monthly/README
    glob_files = str(density_dir / "RHOAnoma_2015*.nc")
    rho_anom = (
        xr.open_mfdataset(glob_files)
        .swap_dims({"k": "Z", "j": "latitude", "i": "longitude"})
        .rename({"Z": "depth", "latitude": "lat", "longitude": "lon"})
        .drop_vars(("i", "j", "k", "timestep", "time_bnds"))
    )
    rho_anom = rho_anom.where(rho_anom != 0)  # zeros indicate no data; replace with nan
    rho_abs = (rhoConst + rho_anom).rename({"RHOAnoma": "rho"})
    rho_abs["rho"].attrs["long_name"] = "Seawater Density"
    rho_abs["rho"].attrs["units"] = "kg m^-3"

    rho_abs.to_netcdf(out_path)

    # clean up temp files
    for f in glob.glob(glob_files):
        os.remove(f)
