import glob
import subprocess
from pathlib import Path

import pandas as pd
import xarray as xr
from tqdm import tqdm


def download_and_process_currents(out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    unmasked_dir = out_dir / "unmasked"
    unmasked_dir.mkdir(exist_ok=True)

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
                unmasked_path = unmasked_dir / filename
                if unmasked_path.exists():
                    continue

                # download the raw data
                raw_out_path = raw_dir / filename
                subprocess.run(
                    [
                        "wget",
                        "-q",
                        "--show-progress",
                        "-O",
                        raw_out_path,
                        url,
                    ]
                )

                # open it up for some editing
                try:
                    ds = xr.open_dataset(raw_out_path)
                except (OSError, ValueError):
                    # file didn't download properly, delet and try again later
                    none_failed = False
                    raw_out_path.unlink()
                    continue

                # swap around some dims, interpolate W to same depth levels as U/V
                ds = ds.swap_dims({"j": "latitude", "i": "longitude"})
                if varname != "WVELMASS":
                    ds = ds.swap_dims({"k": "Z"}).drop_vars("k")
                else:
                    if Z is None:
                        Z = xr.open_dataset(unmasked_dir / "EVEL_2015_01_01.nc").Z
                    ds = (
                        ds.swap_dims({"k_l": "Zl"})
                        .interp(Zl=Z)
                        .drop_vars(("Zl", "k_l"))
                    )
                ds = ds.drop_vars(("i", "j", "timestep", "time_bnds"))
                ds.to_netcdf(unmasked_path)
                raw_out_path.unlink()

        if none_failed:
            print("All files downloaded.")
            raw_dir.rmdir()
            break

    # now we need to compute a land mask.  It isn't included (odd),
    # so all we can do is test where every single file is EXACTLY zero.
    print("Computing land mask...")
    files = list(sorted(Path(f) for f in glob.glob(str(unmasked_dir / "*VEL*.nc"))))
    currents = xr.open_mfdataset(files)
    land = (
        ((currents.NVEL == 0) & (currents.EVEL == 0) & (currents.WVELMASS == 0))
        .all(dim="time")
        .load()
    )
    # set the very bottom layer to land; EVEL and NVEL have data in this layer,
    # but WVELMASS has none.  We want to be consistent.
    land = land.where(land.Z != land.Z[-1], True)

    print("Appling land mask to files...")
    for file in tqdm(files):
        ds = xr.open_dataset(file)
        ds = ds.where(~land)
        ds.to_netcdf(out_dir / file.name)
        file.unlink()

    unmasked_dir.unlink()
