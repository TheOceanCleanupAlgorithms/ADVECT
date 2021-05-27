from pathlib import Path
import pandas as pd
import subprocess
from tqdm import tqdm


def download_ECCO_currents():
    OUT_DIR = Path(__file__).parent / f"ECCO_native/"
    OUT_DIR.mkdir(exist_ok=True)
    dates = pd.date_range("2015", "2016", freq="D", closed="left")
    with tqdm(total=len(dates)*3, desc="DOWNLOADING ECCO CURRENTS", unit="file") as pbar:
        for date in dates: 
            for VARNAME in ["NVEL", "EVEL", "WVELMASS"]:
                filename = f'{VARNAME}_{date.strftime("%Y_%m_%d")}.nc'
                if (OUT_DIR / filename).exists():
                    continue
                url = (
                    f"https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_90/ECCOv4/Release4/nctiles_daily/"
                    f"{VARNAME}/{date.year}/{date.dayofyear:03}/{filename}"
                )
                subprocess.run(
                    args=[
                        "wget",
                        url,
                        "-O",
                        str(OUT_DIR / filename),
                        "--retry-connrefused",
                        "--waitretry=1",
                        "--read-timeout=20",
                        "--timeout=15",
                        "-t 0",
                        "-q",
                    ],
                )
                pbar.update()

    # occasionally you'll get incompletely downloaded files.  Run this command to see if there are any files less than 20 MB
    # find . -name "*.nc" -type 'f' -size -20M
    # then run this command to remove them
    # find . -name "*.nc" -type 'f' -size -20M -delete
    # once removed, run this script again and it will re-download the deleted files

    # you need to download the grid file yourself, it requires an EarthData login.  Just go here:
    # https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/nctiles_grid/ECCO-GRID.nc


if __name__ == "__main__":
    download_ECCO_currents()
