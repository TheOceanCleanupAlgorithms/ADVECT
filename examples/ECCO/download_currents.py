from pathlib import Path
import pandas as pd
import subprocess
from tqdm import tqdm


def download_currents(OUT_DIR: Path):
    while True:
        date_and_varname = []
        for date in pd.date_range("2015", "2016", freq="D", closed="left"):
            for varname in ["NVEL", "EVEL", "WVELMASS"]:
                filename = f'{varname}_{date.strftime("%Y_%m_%d")}.nc'
                if not (OUT_DIR / filename).exists():
                    date_and_varname.append((date, varname, filename))
        for date, varname, filename in tqdm(date_and_varname):
            url = (
                f"https://data.nas.nasa.gov/ecco/download_data.php?file=/eccodata/llc_90/ECCOv4/Release4/nctiles_daily/"
                f"{varname}/{date.year}/{date.dayofyear:03}/{filename}"
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

        # handle corrupted downloads
        find_args = [
            "find",
            str(OUT_DIR),
            "-name",
            '"*.nc"',
            "-type",
            "f",
            "-size",
            "-20M",
        ]
        response = subprocess.getoutput(" ".join(find_args))
        if response == "":
            print("All files downloaded completely!")
            return
        else:
            num_malformed = len(response.split("\n"))
            print(f"Removing {num_malformed} malformed downloads and restarting...")
            print(subprocess.getoutput(" ".join(find_args + ["-delete"])))
