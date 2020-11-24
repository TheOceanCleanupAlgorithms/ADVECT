from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_3d_trajectories(
    outputfile: str,
    current_U_path: str,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Tuple[float, float],
):
    """current_U_path will be used to plot landmass/bathymetry; best if same file as used in creation of outputfile, so
        the boundary behavior matches up.
       lon/lat/depth range should be pretty tight windows.  This plotting is computationally expensive, and if you want
       the result to be at all interactive, this should be a very small window.  It means less generation of grid,
       and only shows particles which enter this domain at some point.  Also, don't use outfiles with too many
       particles."""
    P = xr.open_dataset(outputfile)
    grid = xr.open_dataset(current_U_path).U.isnull().isel(time=0)
    grid = grid.transpose("lon", "lat", "depth").sortby("depth", "ascending")
    smallgrid = grid.sel(
        lon=slice(*lon_range), lat=slice(*lat_range), depth=slice(*depth_range)
    )

    def cell_edges(coord: xr.DataArray):
        depth_bnds = (
            coord - 0.5 * coord.diff(dim=coord.dims[0])
        ).values  # halfway between grid points
        return np.concatenate(
            (
                [2 * depth_bnds[0] - depth_bnds[1]],
                depth_bnds,
                [2 * depth_bnds[-1] - depth_bnds[-2]],
            )
        )  # linearly extrapolate endpoints

    Y, X, Z = np.meshgrid(
        cell_edges(smallgrid.lat),
        cell_edges(smallgrid.lon),
        cell_edges(smallgrid.depth),
    )
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.view_init(15, -80)
    ax.voxels(X, Y, Z, smallgrid.values, edgecolor="k", alpha=.8)

    smallp = P.isel(
        p_id=(
            (P.lon > lon_range[0])
            & (P.lon < lon_range[1])
            & (P.lat > lat_range[0])
            & (P.lat < lat_range[1])
            & (P.depth < depth_range[1])
        ).any(dim="time")
    )

    for i in range(len(smallp.p_id)):
        ax.scatter3D(smallp.isel(p_id=i).lon, smallp.isel(p_id=i).lat, smallp.isel(p_id=i).depth, ".", s=3)
    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)

    ax.set_xlabel('Longitude (deg E)')
    ax.set_ylabel('Latitude (deg N)')
    ax.set_zlabel('Depth (m)')


# cape horn
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/advector_output_2015.nc',
                     current_U_path='../../examples/ECCO/ECCO_interp/U_2015-01-01.nc',
                     lon_range=(-80, -60),
                     lat_range=(-60, -44),
                     depth_range=(-800, -50))
'''
# zanzibar
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/advector_output_2015.nc',
                     current_U_path='../../examples/ECCO/ECCO_interp/U_2015-01-01.nc',
                     lon_range=(35, 50),
                     lat_range=(-10, 10),
                     depth_range=(-150, 0))

# greenland
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/advector_output_2015.nc',
                     current_U_path='../../examples/ECCO/ECCO_interp/U_2015-01-01.nc',
                     lon_range=(-60, -35),
                     lat_range=(55, 70),
                     depth_range=(-1000, 0))

# nz
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/advector_output_2015.nc',
                     current_U_path='../../examples/ECCO/ECCO_interp/U_2015-01-01.nc',
                     lon_range=(160, 180),
                     lat_range=(-50, -30),
                     depth_range=(-500, -10))
'''
