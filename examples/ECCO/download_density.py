import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from get_ECCO_credentials import get_ECCO_credentials


def download_density(out_dir: Path, user: str, password: str):
    for month in range(1, 13):
        url = (
            f"https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/interp_monthly/"
            f"RHOAnoma/2015/RHOAnoma_2015_{month:02}.nc"
        )
        out_path = out_dir / url.split("/")[-1]
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
                    out_path,
                    url,
                ]
            )
