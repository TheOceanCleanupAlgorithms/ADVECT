import subprocess
from pathlib import Path

import pandas as pd
import xarray as xr


def download_and_process_currents(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    Z = None  # placeholder
    while True:
        none_failed = True
        for day in pd.date_range("2015-01-01", "2015-01-31", freq="D"):
            for varname in ("EVEL", "NVEL", "WVELMASS"):
                filename = f"{varname}_{day.strftime('%Y_%m_%d')}.nc"
                url = (
                    f"https://data.nas.nasa.gov/ecco/download_data.php?"
                    f"file=/eccodata/llc_90/ECCOv4/Release4/interp_daily/"
                    f"{varname}/2015/{day.dayofyear:03d}/{filename}"
                )
                out_path = out_dir / filename
                if out_path.exists():
                    continue

                # download the raw data
                temp_out_path = temp_dir / filename
                subprocess.run(
                    [
                        "wget",
                        "-q",
                        "--show-progress",
                        "-O",
                        temp_out_path,
                        url,
                    ]
                )

                # open it up for some editing
                try:
                    ds = xr.open_dataset(temp_out_path)
                except OSError:
                    # file didn't download properly, delet and try again later
                    none_failed = False
                    temp_out_path.unlink()
                    continue

                # swap around some dims, interpolate W to same depth levels as U/V
                ds = ds.swap_dims({"j": "latitude", "i": "longitude"})
                if varname != "WVELMASS":
                    ds = ds.swap_dims({"k": "Z"}).drop_vars("k")
                else:
                    if Z is None:
                        Z = xr.open_dataset(out_dir / "EVEL_2015_01_01.nc").Z
                    ds = (
                        ds.swap_dims({"k_l": "Zl"})
                        .interp(Zl=Z)
                        .drop_vars(("Zl", "k_l"))
                    )
                ds = ds.drop_vars(("i", "j", "timestep", "time_bnds"))
                ds.to_netcdf(out_path)
                temp_out_path.unlink()

        if none_failed:
            print("All files downloaded completely!")
            temp_dir.rmdir()
            return
