from pathlib import Path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from w_integrator import integrator

dataset_path = Path(
    "../examples/HYCOM_GLBy/hycom_GLBy_global_uv_levels_2020-01-01T00.nc"
)
out_path = dataset_path.parent / ("W_calc_from_" + dataset_path.name)


def compute_W():
    UV = (
        xr.open_dataset(dataset_path).squeeze().rename({"water_u": "U", "water_v": "V"})
    )
    UV = UV.assign_coords({"depth": -1 * UV.depth})
    UV = UV.sortby("depth", ascending=True).chunk({"depth": 1})
    # actually, HYCOM depth is the TOP of the grid cell (at least, it must be, because the first value is 0)
    # we need the centers of each grid cell.  So we calculate those here...
    cell_top_depth = UV.depth.values
    cell_widths = np.append(
        np.diff(cell_top_depth)[0], np.diff(cell_top_depth)
    )  # assumes deepest cell is same size as second deepest cell, best we can do
    cell_center_depth = cell_top_depth - cell_widths / 2
    UV = UV.assign_coords({"depth": cell_center_depth})

    W_calc = integrator.generate_vertical_velocity(UV, verbose=True)
    W_calc = W_calc.expand_dims()  # add time back in

    with ProgressBar():
        W_calc.to_netcdf(out_path)


def plot_W(W, level, clip, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(16, 9))
    cf = ax.contourf(
        W.lon,
        W.lat,
        W.isel(depth=level).clip(-clip, clip),
        cmap="bwr",
        levels=30,
        vmin=-clip,
        vmax=clip,
    )
    cbar = plt.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel("W (m/s)")
    plt.title(
        f"HYCOM GLBy Vertical Velocity, 2020-01-01T00, depth={float(W.depth.isel(depth=level)): .0f}m"
    )


W = xr.open_dataarray(out_path, chunks={"depth": 1}).sortby("depth", ascending=False)
W_smooth = W.coarsen(dim={"lat": 15, "lon": 15}, boundary="pad").mean()
with ProgressBar():
    W_smooth.load()

for i in range(10, len(W.depth)):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    plot_W(W_smooth, level=i, clip=1e-3, ax=ax)
    plt.pause(.01)
    input()
    plt.close(fig)
