from typing import Optional, Set, List

import xarray as xr
import glob
import dask
import numpy as np
from dask.diagnostics import ProgressBar


def open_3d_field(
    paths: List[str],
    varnames: Set[str],
    variable_mapping: Optional[dict],
    verbose: bool,
):
    if verbose:
        print("\tOpening netcdf files...")
    concat_dim = next(
        (key for key, value in variable_mapping.items() if value == "time"), "time"
    )
    with ProgressBar():
        vectors = xr.merge(
            (
                xr.open_mfdataset(
                    sorted(glob.glob(path)),
                    data_vars="minimal",
                    parallel=True,
                    concat_dim=concat_dim,
                )
                for path in paths
            ),
            combine_attrs="override",
        )  # use first file's attributes
    vectors = vectors.rename(variable_mapping)
    vectors = vectors[list(varnames)]  # drop any additional variables
    assert set(vectors.dims) == {
        "lat",
        "lon",
        "time",
        "depth",
    }, f"Unexpected/missing dimension(s) ({vectors.dims})"

    # convert longitude [0, 360] --> [-180, 180]
    # this operation could be expensive because of the resorting.  You may want to preprocess your data.
    if max(vectors.lon) > 180:
        print("\tRemapping Longitudes to [-180, 180)...")
        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            vectors["lon"] = ((vectors.lon + 180) % 360) - 180
            vectors = vectors.sortby("lon")

    # convert positive-down depth to positive-up if necessary
    if np.all(vectors.depth >= 0):
        print("\tConverting depth dim to positive-up...")
        vectors["depth"] = -1 * vectors.depth
    if verbose:
        print("\tSorting depth dim...")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        vectors = vectors.sortby(
            "depth", ascending=True
        )  # depth required to be ascending sorted
    return vectors


def open_currents(
    u_path: str,
    v_path: str,
    w_path: str,
    variable_mapping: Optional[dict],
    verbose: bool,
):
    """
    :param u_path: wildcard path to the zonal vector files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_path: wildcard path to the meridional vector files.  See u_path for more details.
    :param w_path: wildcard path to the vertical vector files.  See u_path for more details.
    :param variable_mapping: mapping from names in vector file to advector standard variable names
    """
    return open_3d_field(
        paths=[u_path, v_path, w_path],
        varnames={"U", "V", "W"},
        variable_mapping=variable_mapping,
        verbose=verbose,
    )


def open_seawater_density(
    path: str, variable_mapping: Optional[dict], verbose: bool
) -> xr.Dataset:
    return open_3d_field(
        paths=[path],
        varnames={"rho"},
        variable_mapping=variable_mapping,
        verbose=verbose,
    )


def open_2D_vectorfield(u_path: str, v_path: str, variable_mapping: Optional[dict]):
    """
    :param u_path: wildcard path to the zonal vector files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_path: wildcard path to the meridional vector files.  See u_path for more details.
    :param variable_mapping: mapping from names in vector file to advector standard variable names
    """
    concat_dim = next((key for key, value in variable_mapping.items() if value == "time"), "time")
    with ProgressBar():
        U = xr.open_mfdataset(sorted(glob.glob(u_path)), data_vars="minimal", parallel=True, concat_dim=concat_dim)
        V = xr.open_mfdataset(sorted(glob.glob(v_path)), data_vars="minimal", parallel=True, concat_dim=concat_dim)
        vectors = xr.merge((U, V), combine_attrs="override")  # use first file's attributes
    vectors = vectors.rename(variable_mapping)
    vectors = vectors[["U", "V"]]  # drop any additional variables
    vectors = vectors.squeeze()  # remove any singleton dimensions

    if "depth" in vectors.dims:
        vectors = vectors.sel(depth=0, method="nearest")

    assert set(vectors.dims) == {
        "lat",
        "lon",
        "time",
    }, f"Unexpected/missing dimension(s) ({vectors.dims})"

    # convert longitude [0, 360] --> [-180, 180]
    # this operation could be expensive because of the resorting.  You may want to preprocess your data.
    if max(vectors.lon) > 180:
        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            vectors["lon"] = ((vectors.lon + 180) % 360) - 180
            vectors = vectors.sortby("lon")

    return vectors


def empty_2D_vectorfield():
    return xr.Dataset(
        data_vars={
            "U": (["lat", "lon", "time"], np.ndarray((0, 0, 0))),
            "V": (["lat", "lon", "time"], np.ndarray((0, 0, 0))),
        },
        coords={"lon": [], "lat": [], "time": []},
    )
