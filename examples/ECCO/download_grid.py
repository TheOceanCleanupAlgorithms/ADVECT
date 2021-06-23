import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def download_grid(grid_path: Path, user: str, password: str):
    if grid_path.exists():
        print("Grid file already exists.")
        return
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