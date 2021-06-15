import numpy as np
import xarray as xr


def create_bathymetry_from_land_mask(land_mask: xr.DataArray) -> xr.DataArray:
    """Method: identifies the lower depth bound of the shallowest
        ocean cell (non-null) in each vertical grid column.
    :param land_mask: dimensions {time, depth, lat, lon}, boloean array, True where cell is land"""
    assert np.all(land_mask.depth <= 0), "depth coordinate must be positive up"
    assert np.all(
        np.diff(land_mask.depth) > 0
    ), "depth coordinate must be sorted ascending"

    # In the kernel, particles look up data based on the nearest cell-center.
    # Thus cell bounds are the midpoints between each centers.
    # Very top cell bound is surface, and bottom cell bounds are
    # assumed to be symmetric about bottom cell center.
    depth_diff = np.diff(land_mask.depth)
    depth_bnds = np.concatenate(
        [
            land_mask.depth.values[:1] - depth_diff[0] / 2,
            land_mask.depth.values[:-1] + depth_diff / 2,
            [0],
        ]
    )

    bathy = (
        (~land_mask)
        .assign_coords({"depth": depth_bnds[:-1]})
        .idxmax(dim="depth")
        .where(~land_mask.isel(depth=-1), depth_bnds[-1])
    )

    bathy = bathy.drop(["time", "depth"])
    bathy.name = "bathymetry"
    bathy.attrs = {"units": "m", "positive": "up"}

    return bathy
