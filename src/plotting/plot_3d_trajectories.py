import math
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm

from resources.fetch_etopo1 import fetch_etopo1


def plot_3d_trajectories(
    outputfile: str,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Tuple[float, float],
    stride: Optional[int] = None,
    pan_and_tilt: Optional[Tuple[float, float]] = None,
):
    """current_U_path will be used to plot landmass/bathymetry; best if same file as used in creation of outputfile, so
        the boundary behavior matches up.
       lon/lat/depth range should be pretty tight windows.  This plotting is computationally expensive, and if you want
       the result to be at all interactive, this should be a very small window.  It means less generation of grid,
       and only shows particles which enter this domain at some point.  Also, don't use outfiles with too many
       particles."""
    P = xr.open_dataset(outputfile)
    bathy = fetch_etopo1()
    bathy = bathy.transpose("y", "x")
    smallgrid = bathy.sel(x=slice(*lon_range), y=slice(*lat_range))
    smallgrid['z'] = smallgrid.z.clip(min=depth_range[0], max=0)

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

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    if pan_and_tilt:
        ax.view_init(*pan_and_tilt)
    # ax.voxels(X, Y, Z, smallgrid.values, edgecolor="k", alpha=.8)
    lat_edges = cell_edges(smallgrid.y)
    lon_edges = cell_edges(smallgrid.x)
    ax.set_xlim(lon_edges[0], lon_edges[-1])
    ax.set_ylim(lat_edges[0], lat_edges[-1])
    ax.set_zlim(depth_range[0], 0)
    min_depth = smallgrid.z.min()
    def gridpoint_to_voxel(i, j, stride):
        depth = smallgrid.z.values[j:j+stride, i:i+stride].mean()
        if depth == depth_range[0]:  # leave empty where deeper than min viewing range
            return
        Y, X, Z = np.meshgrid([lat_edges[j], lat_edges[j+stride]],
                              [lon_edges[i], lon_edges[i+stride]],
                              [min_depth, depth])
        ax.voxels(X, Y, Z, np.ones([1, 1, 1]), color='tab:blue', edgecolor='k', linewidth=.1, alpha=.8)

    if not stride:
        stride = math.ceil(len(smallgrid.x) / 40)
    plt.title(f'bathymetry stride = {stride}')
    for i in tqdm(np.arange(len(smallgrid.x) - stride, step=stride)):
        for j in np.arange(len(smallgrid.y) - stride, step=stride):
            gridpoint_to_voxel(i, j, stride)
        #plt.pause(.00001)

    smallp = P.isel(
        p_id=(
            (P.lon > lon_range[0])
            & (P.lon < lon_range[1])
            & (P.lat > lat_range[0])
            & (P.lat < lat_range[1])
            & (P.depth < depth_range[1])
        ).any(dim="time")
    )

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.gca().set_prop_cycle(color=colors[1:])  # first color used for bathy
    for i in range(len(smallp.p_id)):
        ax.scatter3D(smallp.isel(p_id=i).lon, smallp.isel(p_id=i).lat, smallp.isel(p_id=i).depth, ".", s=10,
                     edgecolor='k', linewidth=.2)
    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)

    ax.set_xlabel('Longitude (deg E)')
    ax.set_ylabel('Latitude (deg N)')
    ax.set_zlabel('Depth (m)')

'''
# cape horn
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/neutral/advector_output_2015.nc',
                     lon_range=(-80, -60),
                     lat_range=(-60, -44),
                     depth_range=(-200, -50))

# falkland islands
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/neutral/advector_output_2015.nc',
                     lon_range=(-62, -60),
                     lat_range=(-53, -51),
                     depth_range=(-200, -50),
                     pan_and_tilt=(70, -70),
                     stride=1)

# zanzibar
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/advector_output_2015.nc',
                     current_U_path='../../examples/ECCO/ECCO_interp/U_2015-01-01.nc',
                     lon_range=(35, 50),
                     lat_range=(-10, 10),
                     depth_range=(-150, 0))

# greenland
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/advector_output_2015.nc',
                     lon_range=(-25.1, -24.9),
                     lat_range=(68.3, 68.5),
                     depth_range=(-10000, 0))
'''
# nz
plot_3d_trajectories(outputfile='../../examples/outputfiles/2015_ECCO/neutral/advector_output_2015.nc',
                     lon_range=(162, 182),
                     lat_range=(-50, -32.5),
                     depth_range=(-1000, -30),
                     pan_and_tilt=(60, -160))
