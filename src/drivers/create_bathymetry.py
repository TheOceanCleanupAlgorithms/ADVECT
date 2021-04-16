import xarray as xr
import numpy as np


def create_bathymetry(currents: xr.Dataset) -> xr.Dataset:
    """Creates bathymetry for a current dataset, which encodes its ocean domain.
        Method: identifies the lower depth bound of the shallowest
        ocean cell (non-null) in each vertical grid column."""
    assert np.all(currents.depth <= 0), "depth coordinate must be positive up"
    assert np.all(np.diff(currents.depth) > 0), "depth coordinate must be sorted ascending"

    # In the kernel, particles look up data based on the nearest cell-center.
    # Thus cell bounds are the midpoints between each centers.
    # Very top cell bound is surface, and bottom cell bounds are
    # assumed to be symmetric about bottom cell center.
    depth_diff = np.diff(currents.depth)
    depth_bnds = np.concatenate([
        currents.depth.values[:1] - depth_diff[0] / 2,
        currents.depth.values[:-1] + depth_diff/2,
        [0],
    ])

    first_timestep = currents.isel(time=0)  # only need one timestep
    land = first_timestep.U.isnull() | first_timestep.V.isnull() | first_timestep.W.isnull()

    bathy = (
        (~land)
        .assign_coords({"depth": depth_bnds[:-1]})
        .idxmax(dim="depth")
        .where(~land.isel(depth=-1), depth_bnds[-1])
    )

    bathy = bathy.drop(["time", "depth"])
    bathy.name = "bathymetry"
    bathy.attrs = {"units": "m", "positive": "up"}

    return bathy
