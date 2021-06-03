"""all-in-one script which downloads and interpolates 3D currents and seawater density data."""

import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from download_grid import download_grid
from download_currents import download_currents
from process_currents import interp_currents
from download_density import download_density
from process_density import process_density
from get_ECCO_credentials import get_ECCO_credentials


if __name__ == "__main__":
    user, password = get_ECCO_credentials()

    native_dir = Path("/mnt/advectorbigdata/") / "ECCO_native/"
    grid_path = native_dir / "ECCO-GRID.nc"
    native_dir.mkdir(exist_ok=True)
    print("Downloading native ECCO grid...")
    download_grid(grid_path=grid_path, user=user, password=password)

    print("Downloading ECCO currents on native grid...")
    download_currents(OUT_DIR=native_dir)

    currents_dir = Path(__file__).parent / "currents/"
    currents_dir.mkdir(exist_ok=True)
    resolution_deg = 1
    print(f"Interpolating ECCO currents onto {resolution_deg} degree gaussian grid")
    interp_currents(
        native_dir=native_dir,
        out_dir=currents_dir,
        resolution_deg=resolution_deg,
        grid_path=grid_path,
    )

    print("Removing all files on native grid...")
    shutil.rmtree(native_dir)

    density_dir = Path(__file__).parent / "seawater_density"
    density_dir.mkdir(exist_ok=True)
    density_path = density_dir / "RHO_2015.nc"
    if density_path.exists():
        print(f"{density_path} already exists.  Skipping...")
    else:
        print("Downloading ECCO seawater density on .5 degree gaussian grid...")
        download_density(out_dir=density_dir, user=user, password=password)

        print("Processing raw density files into usable format...")
        process_density(density_dir=density_dir, out_path=density_path)

    print("All Done!")
