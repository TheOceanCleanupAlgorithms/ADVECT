import glob
import gzip
import shutil
import wget
import os
from pathlib import Path
import xarray as xr
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.prepared import prep
import shapely.vectorized
import numpy as np
from tqdm import tqdm


ETOPO_PATH = Path(__file__).parent / "ETOPO1_ICE_SURFACE_GRID.nc"
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/"
    "ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz"
)


def fetch_etopo1() -> xr.Dataset:
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

    # generate land mask from 10m NACIS Natural earth
    etopo1 = xr.open_dataset(ETOPO_PATH)
    if 'is_land' not in etopo1.variables:
        print('Downloading NACIS Natural Earth Shapefiles...')
        land_shp_fname = shpreader.natural_earth(resolution='10m',
                                                 category='physical', name='land')
        land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
        land = prep(land_geom)

        print('Generating land mask.  This could take a couple minutes.')
        X, Y = np.meshgrid(etopo1.x, etopo1.y)
        land_mask = np.concatenate([
            shapely.vectorized.contains(land, x, y)
            for x, y in tqdm(zip(np.array_split(X, 50), np.array_split(Y, 50)), total=50)
        ])

        print('Writing land mask to disk...')
        etopo1 = etopo1.assign({'is_land': (('y', 'x'), land_mask)})
        etopo1.load()  # gonna need the data in ram, else data lost in self-overwrite
        etopo1.to_netcdf(ETOPO_PATH)

    return etopo1


if __name__ == "__main__":
    fetch_etopo1()


'''
show the importance of this land mask versus just using land=above sea level
import matplotlib.pyplot as plt
bathy = fetch_etopo1()
mask = bathy.is_land.values.astype('int')
mask[(bathy.is_land.values & (bathy.z.values < 0))] = 2
plt.imshow(mask, origin='lower', cmap='viridis')
cbar = plt.colorbar(fraction=.04)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['ocean\n(NACIS Natural Earth)', 'land\n(NACIS Natural Earth)', 'naive ocean\n(ETOPO1.z < 0)'])
'''
