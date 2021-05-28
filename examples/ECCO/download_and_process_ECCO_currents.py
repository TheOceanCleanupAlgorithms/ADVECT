import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from download_ECCO_currents import download_ECCO_currents, download_ECCO_grid
from interp_ECCO_to_latlon import interp_ECCO_currents

if __name__ == "__main__":
    native_grid_dir = Path(__file__).parent / "ECCO_native/"
    native_grid_dir.mkdir(exist_ok=True)
    print("Directing you to the ECCO grid download...")
    download_ECCO_grid(OUT_DIR=native_grid_dir)
    print("Downloading ECCO currents on native grid...")
    download_ECCO_currents(OUT_DIR=native_grid_dir)

    interp_grid_dir = Path(__file__).parent / "ECCO_interp/"
    interp_grid_dir.mkdir(exist_ok=True)
    resolution_deg = 1
    print(f"Interpolating ECCO to {resolution_deg} degree gaussian grid")
    interp_ECCO_currents(
        native_grid_dir=native_grid_dir,
        interp_grid_dir=interp_grid_dir,
        resolution_deg=resolution_deg,
    )

    print("Removing current files on native grid...")
    shutil.rmtree(native_grid_dir)

    print("All Done!")
