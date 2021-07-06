import glob
from typing import Optional, Set, List, Callable

import dask
import numpy as np
import xarray as xr

from io_tools.create_bathymetry import create_bathymetry_from_land_mask


def open_3d_currents(
    u_path: str,
    v_path: str,
    w_path: str,
    preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]],
):
    """
    :param u_path: wildcard path to the zonal vector files.  Fed to glob.glob.
    :param v_path: wildcard path to the meridional vector files.
    :param w_path: wildcard path to the vertical vector files.
    :param preprocessor: func to call on the xarray dataset to perform operation before loading in advector, such as renaming variables.
    """
    currents = open_vectorfield(
        paths=[u_path, v_path, w_path],
        varnames={"U", "V", "W"},
        keep_depth_dim=True,
        preprocessor=preprocessor,
    )
    # encode the model domain, taken as where all the current components are non-null, as bathymetry
    print("Calculating bathymetry of current dataset...")
    first_timestep = currents.isel(time=0)  # only need one timestep
    land_mask = (
        first_timestep.U.isnull()
        | first_timestep.V.isnull()
        | first_timestep.W.isnull()
    )

    return xr.merge(
        (currents, create_bathymetry_from_land_mask(land_mask)),
        combine_attrs="override",
    )


def open_2d_currents(
    u_path: str,
    v_path: str,
    preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]],
):
    """
    :param u_path: wildcard path to the zonal vector files.  Fed to glob.glob.
    :param v_path: wildcard path to the meridional vector files.
    :param preprocessor: func to call on the xarray dataset to perform operation before loading in advector, such as renaming variables.
    """
    return open_vectorfield(
        paths=[u_path, v_path],
        varnames={"U", "V"},
        keep_depth_dim=False,
        preprocessor=preprocessor,
    )


def open_seawater_density(
    path: str,
    preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]],
) -> xr.Dataset:
    """
    :param path: wildcard path to the seawater density files.  Fed to glob.glob.
    :param preprocessor: func to call on the xarray dataset to perform operation before loading in advector, such as renaming variables.
    """
    return open_vectorfield(
        paths=[path], varnames={"rho"}, keep_depth_dim=True, preprocessor=preprocessor
    )


def open_wind(
    u_path: str, v_path: str, preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]]
):
    """
    :param u_path: wildcard path to the zonal vector files.  Fed to glob.glob.
    :param v_path: wildcard path to the meridional vector files.
    :param preprocessor: func to call on the xarray dataset to perform operation before loading in advector, such as renaming variables.
    """
    return open_vectorfield(
        paths=[u_path, v_path],
        varnames={"U", "V"},
        keep_depth_dim=False,
        preprocessor=preprocessor,
    )


def open_vectorfield(
    paths: List[str],
    varnames: Set[str],
    keep_depth_dim: bool,
    preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]],
) -> xr.Dataset:
    print("\tOpening NetCDF files...")
    vectors = xr.merge(
        (
            xr.open_mfdataset(
                sorted(glob.glob(path)),
                data_vars="minimal",
                parallel=True,
            )
            for path in paths
        ),
        combine_attrs="override",  # use first file's attributes
    )
    if preprocessor is not None:
        vectors = preprocessor(vectors)

    vectors = vectors[list(varnames)]  # drop any additional variables

    if keep_depth_dim:
        # convert positive-down depth to positive-up if necessary
        if np.all(vectors.depth >= 0):
            print("\tConverting depth to positive-up...")
            vectors["depth"] = -1 * vectors.depth
        if not np.all(np.diff(vectors.depth) >= 0):
            print("\tDepth dimension not sorted.  Sorting..")
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                vectors = vectors.sortby("depth", ascending=True)
        expected_dims = {"lat", "lon", "time", "depth"}
    else:
        if "depth" in vectors.dims:
            print("\tExtracting nearest level to depth=0...")
            vectors = vectors.sel(depth=0, method="nearest")
        expected_dims = {"lat", "lon", "time"}
    assert (
        set(vectors.dims) == expected_dims
    ), f"Unexpected/missing dimension(s) (Found : {set(vectors.dims)}, Expected : {expected_dims})"

    if max(vectors.lon) > 180:
        print("\tRolling longitude domain from [0, 360) to [-180, 180).")
        print(
            "\tThis operation is expensive.  You may want to preprocess your data to the correct domain."
        )
        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            vectors["lon"] = ((vectors.lon + 180) % 360) - 180
            vectors = vectors.sortby("lon")

    return vectors
