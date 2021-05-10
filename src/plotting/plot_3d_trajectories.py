from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

from io_tools.create_bathymetry import create_bathymetry_from_land_mask


class BathyPlotType(Enum):
    gridded = 0
    contour = 1


def plot_3d_trajectories(
    particles: xr.Dataset,
    land_mask: xr.DataArray,
    bathymetry_plot_type: BathyPlotType = BathyPlotType.gridded,
):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")
    ax.view_init(15, -80)
    if bathymetry_plot_type == BathyPlotType.gridded:
        plot_gridded_bathymetry(ax=ax, land_mask=land_mask)
    elif bathymetry_plot_type == BathyPlotType.contour:
        plot_contour_bathymetry(ax=ax, land_mask=land_mask)
    else:
        raise ValueError("Invalid BathyPlotType")

    for i in range(len(particles.p_id)):
        ax.scatter3D(
            particles.isel(p_id=i).lon,
            particles.isel(p_id=i).lat,
            particles.isel(p_id=i).depth,
            ".",
            s=3,
        )
    ax.set_xlim([land_mask.lon.min(), land_mask.lon.max()])
    ax.set_ylim([land_mask.lat.min(), land_mask.lat.max()])
    ax.set_zlim([min(land_mask.depth), -1*(min(land_mask.depth))])
    ax.set_xlabel('Longitude (deg E)')
    ax.set_ylabel('Latitude (deg N)')
    ax.set_zlabel('Depth (m)')


def plot_gridded_bathymetry(
    ax: plt.Axes,
    land_mask: xr.DataArray,
):
    bathy = create_bathymetry_from_land_mask(land_mask=land_mask)
    floor = np.min(bathy.values) - 1
    X, Y = np.meshgrid(bathy.lon, bathy.lat)
    dx = np.diff(bathy.lon)[0]
    dy = np.diff(bathy.lat)[0]
    facecolors = np.empty((len(X.ravel()), 6), dtype=object)
    facecolors[:] = "tab:blue"
    facecolors[bathy.values.ravel() == 0, 1] = "tab:green"

    ax.bar3d(X.ravel(), Y.ravel(), floor, dx, dy, bathy.values.ravel()-floor,
             edgecolor="k", linewidth=.1, alpha=.8, color=facecolors.ravel())


def plot_contour_bathymetry(
    ax: plt.Axes,
    land_mask: xr.DataArray,
):
    bathy = create_bathymetry_from_land_mask(land_mask=land_mask)
    X, Y = np.meshgrid(bathy.lon, bathy.lat)
    color = np.zeros_like(bathy)
    color[bathy.values == 0] = 1
    cmap = colors.ListedColormap(['tab:blue', 'tab:green'])

    my_col = cmap(color)
    ax.plot_surface(X=X, Y=Y, Z=bathy.values, alpha=.9, facecolors=my_col)


if __name__ == "__main__":
    # example plotting around cape horn
    lon_range = (-80, -40)
    lat_range = (-70, -30)
    depth_range = (-10000, 0)
    P = xr.open_dataset(
        '../../examples/outputfiles/ECCO_2015_3D/3D_uniform_source_2015/ADVECTOR_3D_output_2015.nc',
    )
    P = P.isel(
        p_id=(
            (P.lon >= lon_range[0]) &
            (P.lon <= lon_range[1]) &
            (P.lat >= lat_range[0]) &
            (P.lat <= lat_range[1]) &
            (P.depth >= depth_range[0]) &
            (P.depth <= depth_range[1])
        ).any(dim="time")
    )
    ECCO_land_mask = (
        xr.open_dataset('../../examples/ECCO/ECCO_interp/U_2015-01-01.nc')
        .squeeze().U.isnull()
        .sortby("depth")
        .sel(
            lon=slice(*lon_range),
            lat=slice(*lat_range),
        )
    )
    plot_3d_trajectories(
        particles=P,
        land_mask=ECCO_land_mask,
        bathymetry_plot_type=BathyPlotType.contour,
    )
