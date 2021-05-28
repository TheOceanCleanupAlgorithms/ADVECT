import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from get_ECCO_credentials import get_ECCO_credentials


def download_ECCO_grid(OUT_DIR: Path):
    grid_path = OUT_DIR / "ECCO-GRID.nc"
    if grid_path.exists():
        print("Grid file already exists.")
        return
    user, password = get_ECCO_credentials()
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
            str(grid_path),
            "https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/nctiles_grid/ECCO-GRID.nc",
        ]
    )
