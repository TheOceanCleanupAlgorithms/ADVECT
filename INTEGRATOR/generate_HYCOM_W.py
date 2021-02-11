import os
from pathlib import Path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from INTEGRATOR import integrator, generate_ECCO_W


def compute_W(hycom_path, out_path, coarsen=True):
    UV = (
        xr.open_dataset(hycom_path).squeeze().rename({"water_u": "U", "water_v": "V"})
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

    if coarsen:
        print('Coarsening UV...')
        UV = UV.coarsen(dim={"lat": 50, "lon": 25}, boundary="pad").mean()
        UV = UV.interp(lat=np.arange(-80, 81), lon=np.arange(0, 360))
        with ProgressBar():
            UV.load()

    W_calc = integrator.generate_vertical_velocity(UV, verbose=False)
    W_calc = W_calc.expand_dims()  # add time back in

    encoding = {'W': {'_FillValue': -30000, 'scale_factor': 0.000001, 'dtype': np.int16}}

    with ProgressBar():
        W_calc.to_netcdf(out_path, encoding=encoding)


def plot_W(W, level, clip=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(16, 9))
    if clip is None:
        clip = float(np.abs(W.isel(depth=level)).max())
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
    plt.title(f"Vertical Velocity, depth={float(W.depth.isel(depth=level)): .0f}m")


dataset_path = Path(
    "../examples/HYCOM_GLBy/hycom_GLBy_global_uv_levels_2020-01-01T00.nc"
)
hycom_w_out_path = dataset_path.parent / ("W_calc_from_" + dataset_path.name)

if not os.path.exists(hycom_w_out_path):
    print('computing hycom vertical velocity...')
    compute_W(dataset_path, hycom_w_out_path, coarsen=True)

print('loading & coarsening hycom vertical velocity...')
HYCOM_W = xr.open_dataarray(hycom_w_out_path, chunks={"depth": 1}).sortby("depth", ascending=False)
HYCOM_W_coarse = HYCOM_W.coarsen(dim={"lat": 50, "lon": 25}, boundary="pad").mean()
with ProgressBar():
    HYCOM_W_coarse.load()

print('loading & coarsening hycom horizontal velocities...')
HYCOM_UV = xr.open_dataset(dataset_path, chunks={"depth": 1}).squeeze().rename({"water_u": "U", "water_v": "V"})
HYCOM_UV['depth'] = -1 * HYCOM_UV.depth
HYCOM_UV_coarse = HYCOM_UV.coarsen(dim={"lat": 50, "lon": 25}, boundary="pad").mean()
with ProgressBar():
    HYCOM_UV_coarse.load()

print('loading ECCO horizontal velocities & calculating ecco vertical velocity...')
ECCO_UV, ECCO_W_true = generate_ECCO_W.load_ECCO()
ECCO_W = generate_ECCO_W.calculate_W(ECCO_UV).load()
ECCO_UV = ECCO_UV.sortby('depth', ascending=False)


def plot_W_levels(W, clip=None):
    for i in range(len(W.depth)):
        fig, ax = plt.subplots(1, figsize=(16, 9))
        plot_W(W, level=i, clip=clip, ax=ax)
        plt.pause(.01)
        input()
        plt.close(fig)
# plot_W_levels(HYCOM_W_coarse, clip=None)  # uncomment for visual look; add clip to fix colorbar extent


def hycom_vs_ecco_current_speed(ECCO, HYCOM):
    """compute histograms of raw current speed at each depth level; plot"""
    ECCO_spd_median = np.nanmedian(np.sqrt(ECCO.U ** 2 + ECCO.V ** 2).values.reshape((len(ECCO.depth), -1)), axis=1)
    HYCOM_spd_median = np.nanmedian(np.sqrt(HYCOM.U ** 2 + HYCOM.V ** 2).values.reshape((len(HYCOM.depth), -1)), axis=1)

    plt.figure()
    plt.plot(ECCO_spd_median, ECCO.depth, label='ecco 2015-01-01')
    plt.plot(HYCOM_spd_median, HYCOM.depth, label='hycom 2020-01-01')
    plt.xlabel("median current speed at depth levels (m/s)")
    plt.ylabel("depth (m)")
    plt.legend()
    plt.title("ECCO vs HYCOM Current Speed Profile")
# hycom_vs_ecco_current_speed(ECCO_UV, HYCOM_UV_coarse)


def plot_profile_comparisons_between_lats(lat_bnds: tuple):
    HYCOM_prof = HYCOM_W_coarse.sel(lat=slice(*lat_bnds)).median(dim=['lat', 'lon'])
    ECCO_prof = ECCO_W.sel(lat=slice(*lat_bnds)).median(dim=['lat', 'lon'])
    ECCO_prof_true = ECCO_W_true.sel(lat=slice(*lat_bnds)).median(dim=['lat', 'lon'])

    plt.figure()
    plt.plot(HYCOM_prof, HYCOM_prof.depth, label='hycom')
    plt.plot(ECCO_prof, ECCO_prof.depth, label='ecco calc')
    plt.plot(ECCO_prof_true, ECCO_prof_true.depth, label='ecco true')
    plt.xlabel("W (m/s)")
    plt.ylabel("depth (m)")
    plt.legend()
    plt.title(f"ECCO vs HYCOM median W, lat={lat_bnds}")

# plot_profile_comparisons_between_lats((-5, 5))  # tropics
# plot_profile_comparisons_between_lats((40, 50))  # midlats NH
# plot_profile_comparisons_between_lats((-50, -40)) # midlats SH


def compare_horizontals(var='U'):
    """does a comparison of horizontal velocities between ecco and hycom
    :param var: one of {'U', 'V'}
    """
    for depth in HYCOM_UV.depth:
        integrator.compare_Ws(('ECCO '+var, ECCO_UV[var]), ('HYCOM '+var, HYCOM_UV_coarse[var].roll(lon=len(HYCOM_UV_coarse.lon)//2, roll_coords=False)), depth=depth, clip=2)
        plt.pause(.01)
        input()
        plt.close()


def compare_vertical():
    """does a comparison of vertical velocities between ecco and hycom
    """
    for depth in HYCOM_W_coarse.depth:
        integrator.compare_Ws(('ECCO W', ECCO_W), ('HYCOM W', HYCOM_W_coarse.roll(lon=len(HYCOM_UV_coarse.lon)//2, roll_coords=False)),
                              depth=depth, clip=5e-4)
        plt.pause(.01)
        input()
        plt.close()


# compare_horizontals('U')
# compare_horizontals('V')


def vertical_profiles_spaghet(lat=0):
    fig, ax = plt.subplots(1, 2, sharex='all', sharey='all')
    ax[0].plot(HYCOM_W_coarse.sel(lat=lat, method='nearest').transpose('depth', 'lon'), HYCOM_W_coarse.depth, alpha=.1)
    ax[0].set_title('HYCOM')
    ax[1].plot(ECCO_W.sel(lat=lat, method='nearest').transpose('depth', 'lon'), ECCO_W.depth, alpha=.1)
    ax[1].set_title('ECCO')
    ax[0].set_xlabel('W (m/s)')
    ax[1].set_xlabel('W (m/s)')
    ax[0].set_ylabel('depth (m)')
    ax[0].set_xlim(-max(np.abs(ax[0].get_xlim())), max(np.abs(ax[0].get_xlim())))  # aka center x=0
    fig.suptitle(f'Vertical Profiles of Vertical Velocity at Latitude={lat} N')
