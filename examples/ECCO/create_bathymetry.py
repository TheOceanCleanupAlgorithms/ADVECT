import xarray as xr
import numpy as np


def create_ECCO_bathymetry():
    """Creates a bathymetry file for the ECCO dataset.
        Method: identifies the upper depth bound of the shallowest
        null cell in each vertical grid column."""
    U = xr.open_dataset("./ECCO_interp/U_2015-01-01.nc")["U"].squeeze()

    # compute depth bounds symmetrically about each cell-center,
    # with initial boundary condition depth_bnds[0] = 0
    depth_bnds = np.zeros(len(U.depth) + 1)
    for i in range(1, len(depth_bnds)):
        depth_bnds[i] = 2 * U.depth[i - 1] - depth_bnds[i - 1]

    bathy = (
        (~U.isnull())
        .assign_coords({"depth": depth_bnds[:-1]})
        .idxmin(dim="depth")
        .where(U.isel(depth=-1).isnull(), depth_bnds[-1])
    )

    bathy = bathy.drop(["time", "depth"])
    bathy.name = "bathymetry"
    bathy.attrs = {"units": "m", "positive": "up"}

    bathy.to_netcdf("./ECCO_interp/bathymetry.nc")


if __name__ == "__main__":
    create_ECCO_bathymetry()
