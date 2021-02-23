"""
Compare ECCO vs GLORYS vs HYCOM
"""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from INTEGRATOR import integrator


def compare_ECCO_GLORYS():
    ECCO, ECCO_W = load_ECCO()
    GLORYS, GLORYS_W, GLORYS_W_coarse = load_GLORYS()

    compare_Ws(ECCO_W, "ECCO_W", GLORYS_W_coarse, "GLORYS_W_coarse")


def compare_ECCO_HYCOM():
    ECCO, ECCO_W = load_ECCO()
    HYCOM, HYCOM_W, HYCOM_W_coarse = load_HYCOM()

    compare_Ws(ECCO_W, "ECCO_W", HYCOM_W_coarse, "HYCOM_W_coarse")


def compare_HYCOM_GLORYS():
    HYCOM, HYCOM_W, HYCOM_W_coarse = load_HYCOM()
    GLORYS, GLORYS_W, GLORYS_W_coarse = load_GLORYS()

    compare_Ws(HYCOM_W_coarse, "HYCOM_W_coarse", GLORYS_W_coarse, "GLORYS_W_coarse")


def compare_Ws(W1, name1, W2, name2):
    for depth in reversed(W1.depth):
        integrator.compare_Ws(
            (name1, W1), (name2, W2), depth=depth
        )
        plt.pause(0.01)
        input()
        plt.close()


def load_ECCO():
    ECCO = (
        xr.merge(
            (
                xr.open_dataset("../../examples/ECCO/ECCO_interp/U_2015-01-01.nc"),
                xr.open_dataset("../../examples/ECCO/ECCO_interp/V_2015-01-01.nc"),
                xr.open_dataset("../../examples/ECCO/ECCO_interp/W_2015-01-01.nc"),
            )
        )
        .squeeze()
        .sortby("depth")
    )
    ECCO_W = integrator.generate_vertical_velocity(ECCO).load()
    return ECCO, ECCO_W


def load_GLORYS():
    GLORYS = (
        xr.merge(
            (
                xr.open_dataset("../../examples/GLORYS/GLORYS_2015-01-01_U.nc"),
                xr.open_dataset("../../examples/GLORYS/GLORYS_2015-01-01_V.nc"),
            )
        )
        .squeeze()
        .rename({"uo": "U", "vo": "V", "latitude": "lat", "longitude": "lon"})
    )
    GLORYS = GLORYS.assign_coords({"depth": -1 * GLORYS.depth}).sortby("depth")
    GLORYS_W = integrator.generate_vertical_velocity(GLORYS)

    GLORYS_W_fname = "../../examples/GLORYS/GLORYS_2015-01-01_W.nc"
    if not os.path.exists(GLORYS_W_fname):
        with ProgressBar():
            print("Computing GLORYS_W...")
            GLORYS_W.to_netcdf(GLORYS_W_fname)

    with ProgressBar():
        print("Loading GLORYS_W from disk...")
        GLORYS_W = xr.open_dataarray(GLORYS_W_fname).load()
        print("Coarsening GLORYS_W to 2 deg...")
        GLORYS_W_coarse = GLORYS_W.coarsen(
            dim={"lat": 24, "lon": 24}, boundary="pad"
        ).mean()

    return GLORYS, GLORYS_W, GLORYS_W_coarse


def load_HYCOM():
    processed_fname = (
        "../../examples/HYCOM_GLBv/hycom_GLBv_global_uv_2015-01-01T12_processed.nc"
    )
    if not os.path.exists(processed_fname):
        HYCOM = xr.open_dataset(
            "../../examples/HYCOM_GLBv/hycom_GLBv_global_uv_2015-01-01T12.nc"
        )
        # this dataset has an unequally spaced latitude grid
        HYCOM = HYCOM.chunk({"depth": 1}).interp(
            lat=np.arange(HYCOM.lat.min(), HYCOM.lat.max() + 1 / 12, 1 / 12)
        )

        HYCOM = HYCOM.assign_coords({"depth": -1 * HYCOM.depth}).sortby("depth")

        # actually, HYCOM depth is the TOP of the grid cell (at least, it must be, because the first value is 0)
        # we need the centers of each grid cell.  So we calculate those here...
        cell_top_depth = HYCOM.depth.values
        cell_widths = np.append(
            np.diff(cell_top_depth)[0], np.diff(cell_top_depth)
        )  # assumes deepest cell is same size as second deepest cell, best we can do
        cell_center_depth = cell_top_depth - cell_widths / 2
        HYCOM = HYCOM.assign_coords({"depth": cell_center_depth})

        with ProgressBar():
            print("Interpolating original HYCOM dataset to regular lat grid...")
            HYCOM.to_netcdf(processed_fname)

    HYCOM = (
        xr.open_dataset(processed_fname, lock=False)
        .rename({"water_u": "U", "water_v": "V"})
        .squeeze()
    )

    HYCOM_W = integrator.generate_vertical_velocity(HYCOM.chunk({"depth": 1, "lon": len(HYCOM.lon)//5}), auto_chunk=False)

    HYCOM_W_fname = "../../examples/HYCOM_GLBv/hycom_GLBv_global_w_2015-01-01T12.nc"
    if not os.path.exists(HYCOM_W_fname):
        with ProgressBar():
            print("Computing HYCOM_W...")
            HYCOM_W.to_netcdf(HYCOM_W_fname)

    with ProgressBar():
        print("Loading HYCOM_W from disk...")
        HYCOM_W = xr.open_dataarray(HYCOM_W_fname).load()
        print("Coarsening HYCOM_W to 2 deg...")
        HYCOM_W_coarse = HYCOM_W.coarsen(
            dim={"lat": 24, "lon": 24}, boundary="pad"
        ).mean()

    return HYCOM, HYCOM_W, HYCOM_W_coarse


def plot_speed_vertical_profile(UV, name):
    spd = (UV.U**2 + UV.V**2)**.5
    prof = spd.mean(dim=['lat', 'lon'])
    plt.plot(prof, prof.depth, label=name)