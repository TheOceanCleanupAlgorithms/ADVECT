from pathlib import Path
import pandas as pd
import subprocess
import webbrowser
from tqdm import tqdm


def download_ECCO_grid():
    OUT_DIR = Path(__file__).parent / "ECCO_native/"
    OUT_DIR.mkdir(exist_ok=True)
    if (OUT_DIR / "ECCO-GRID.nc").exists():
        print("Grid file already exists.")
        return
    print(
        "You need to download the grid file yourself, "
        "as it requires an EarthData login; you may need to create one.\n"
        "Place the file at examples/ECCO/ECCO_NATIVE/ECCO-GRID.nc."
    )
    webbrowser.open(
        "https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/nctiles_grid/ECCO-GRID.nc"
    )


def download_ECCO_currents():
    OUT_DIR = Path(__file__).parent / "ECCO_native/"
    OUT_DIR.mkdir(exist_ok=True)
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
