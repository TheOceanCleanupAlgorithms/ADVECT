from typing import Callable, Optional

import xarray as xr
import glob
import dask
import numpy as np


def open_netcdf_vectorfield(u_path: str, v_path: str, preprocessor: Optional[Callable]):
    """
    :param u_path: wildcard path to the zonal vector files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_path: wildcard path to the meridional vector files.  See u_path for more details.
    :param preprocessor: func to call on the xarray dataset to perform operation before loading in advector, such as renaming variables.
    """
    U = xr.open_mfdataset(sorted(glob.glob(u_path)), data_vars="minimal", chunks="auto")
    V = xr.open_mfdataset(sorted(glob.glob(v_path)), data_vars="minimal", chunks="auto")
    vectors = xr.merge((U, V))

    # Apply preprocessor before doing anything else.
    if preprocessor is not None:
        vectors = preprocessor(vectors)

    vectors = vectors[['U', 'V']]  # drop any additional variables
    vectors = vectors.squeeze()  # remove any singleton dimensions
    

    if "depth" in vectors.dims:
        vectors = vectors.sel(depth=0, method='nearest')

    assert set(vectors.dims) == {'lat', 'lon', 'time'}, f"Unexpected/missing dimension(s) ({vectors.dims})"

    # convert longitude [0, 360] --> [-180, 180]
    # this operation could be expensive because of the resorting.  You may want to preprocess your data.
    if max(vectors.lon) > 180:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            vectors['lon'] = ((vectors.lon + 180) % 360) - 180
            vectors = vectors.sortby('lon')

    return vectors


def empty_vectorfield():
    return xr.Dataset(
        data_vars={
            "U": (["lat", "lon", "time"], np.ndarray((0, 0, 0))),
            "V": (["lat", "lon", "time"], np.ndarray((0, 0, 0))),
        },
        coords={"lon": [], "lat": [], "time": []},
    )
