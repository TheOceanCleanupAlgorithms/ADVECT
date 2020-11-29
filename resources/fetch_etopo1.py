import glob
import gzip
import shutil
import wget
import os
from pathlib import Path


ETOPO_PATH = Path(__file__).parent / "ETOPO1_ICE_SURFACE_GRID.nc"
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/"
    "ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz"
)


def fetch_etopo1() -> Path:
    """returns path to the ETOPO1 file.  If doesn't exist, then downloads it."""
    if not ETOPO_PATH.exists():
        print("Downloading ETOPO1...")
        zipped = ETOPO_PATH.with_suffix(".gz")

        def bar_custom(current, total, width=80):
            print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total), end='\r')
        wget.download(ETOPO_URL, str(zipped), bar=bar_custom)

        print("\nUnzipping ETOPO1...")
        with gzip.open(zipped, "rb") as f_in:
            with open(ETOPO_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print("Cleaning up...")
        os.remove(zipped)
        for tmp in glob.glob(str(ETOPO_PATH.parent / "*.tmp")):
            os.remove(tmp)

    return ETOPO_PATH


if __name__ == "__main__":
    fetch_etopo1()
