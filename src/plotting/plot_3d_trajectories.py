from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_3d_trajectories(
    particles: xr.Dataset,
    land_mask: xr.DataArray,
):
    land_mask = land_mask.transpose("lon", "lat", "depth").sortby("depth", "ascending")


    def cell_edges(coord: xr.DataArray):
        depth_bnds = (
            coord - 0.5 * coord.diff(dim=coord.dims[0])
        ).values  # halfway between grid points
        return np.concatenate(
            (
                [coord[0] - (coord[1]-coord[0])/2],
                depth_bnds,
                [coord[-1] - (coord[-2]-coord[-1])/2],
            )
        )  # linearly extrapolate endpoints

    Y, X, Z = np.meshgrid(
        cell_edges(land_mask.lat),
        cell_edges(land_mask.lon),
        cell_edges(land_mask.depth),
    )
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.view_init(15, -80)
    ax.voxels(X, Y, Z, land_mask.values, edgecolor="k", alpha=.8)
    for i in range(len(particles.p_id)):
        ax.scatter3D(
            particles.isel(p_id=i).lon,
            particles.isel(p_id=i).lat,
            particles.isel(p_id=i).depth,
            ".",
            s=3,
        )
    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)

    ax.set_xlabel('Longitude (deg E)')
    ax.set_ylabel('Latitude (deg N)')
    ax.set_zlabel('Depth (m)')


if __name__ == "__main__":
    # cape horn
    particles = xr.open_dataset('../../examples/outputfiles/ECCO_2015_3D/3D_uniform_source_2015/ADVECTOR_3D_output_2015.nc',)
    lon_range = (-80, -60)
    lat_range = (-56, -46)
    depth_range = (-10000, 0)
    particles = particles.isel(
        p_id=(
            (particles.lon >= lon_range[0]) &
            (particles.lon <= lon_range[1]) &
            (particles.lat >= lat_range[0]) &
            (particles.lat <= lat_range[1]) &
            (particles.depth >= depth_range[0]) &
            (particles.depth <= depth_range[1])
        ).any(dim="time")
    )
    land_mask = xr.open_dataset('../../examples/ECCO/ECCO_interp/U_2015-01-01.nc').squeeze().U.isnull().sortby("depth")
    land_mask = land_mask.sel(
        lon=slice(*lon_range), lat=slice(*lat_range), depth=slice(*depth_range)
    )
    plot_3d_trajectories(
        particles=particles,
        land_mask=land_mask,
    )
