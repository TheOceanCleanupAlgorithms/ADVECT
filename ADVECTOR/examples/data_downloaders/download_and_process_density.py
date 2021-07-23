import subprocess
from pathlib import Path

import xarray as xr


def download_and_process_density(out_path: Path, user: str, password: str):
    temp_dir = out_path.parent / "density_temp"
    temp_dir.mkdir(exist_ok=True)
    url = (
        f"https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/interp_monthly/"
        f"RHOAnoma/2015/RHOAnoma_2015_01.nc"
    )
    tmp_out_path = temp_dir / url.split("/")[-1]
    if not out_path.exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "--show-progress",
                "--user",
                user,
                "--password",
                password,
                "-O",
                tmp_out_path,
                url,
            ]
        )

    rhoConst = 1029  # source: https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/nctiles_monthly/README
    rho_anom = (
        xr.open_dataset(tmp_out_path)
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
    tmp_out_path.unlink()
    temp_dir.rmdir()
