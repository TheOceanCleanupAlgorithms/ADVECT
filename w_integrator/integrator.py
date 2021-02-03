"""
Step 1: calculate horizontal velocities at the boundaries between cells (for whole dataset!)
Step 2: do a 2x2 convolution on this to get the horizontal mass flux into every cell
Step 3: vertically integrate
"""
from typing import Tuple

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def generate_vertical_velocity(UV, verbose=False):
    # Step 1
    if verbose:
        print("calculating velocities at the boundaries between cells")
    # first, generate new grids for the edges of cells
    lat_bnds, lon_bnds = [
        np.concatenate([coord[:1]-np.diff(coord)[0]/2, coord[:-1] + np.diff(coord)/2, coord[-1:]+np.diff(coord)[-1]/2])
        for coord in (UV.lat.values, UV.lon.values)
    ]  # first and last bnd are the same point for lon_bnds

    z_bnds = np.zeros(len(UV.depth) + 1)
    for i in reversed(range(len(z_bnds) - 1)):
        z_bnds[i] = (float(2*UV.depth[i] - z_bnds[i+1]))

    # coordinates of V_at_lat_bnds: (depth, lat_bnds, lon)
    V_at_lat_bnds = UV.V.interp(lat=lat_bnds).rename({'lat': 'lat_bnds'})
    V_at_lat_bnds[:, 0, :] = 0  # zero meridional flow at the poles?  Undefined, really, but w/e
    V_at_lat_bnds[:, -1, :] = 0

    U_at_lon_bnds = UV.U.interp(lon=lon_bnds).rename({'lon': 'lon_bnds'})
    U_at_lon_bnds[:, :, 0] = UV.U.isel({'lon': [0, -1]}).mean(dim='lon')  # calculate the meridian wrap
    U_at_lon_bnds[:, :, -1] = U_at_lon_bnds[:, :, 0]

    # Step 2
    if verbose:
        print('Calculating mass flux into every cell due to U/V...')
    # we offset the bounds arrays a bit so that we wind up with things the same shape as the cell grid; each value
    # thus corresponds to the flux across the west (e.g.) boundary of a particular grid cell
    U_w = U_at_lon_bnds.values[:, :, :-1]
    U_e = U_at_lon_bnds.values[:, :, 1:]
    V_s = V_at_lat_bnds.values[:, :-1]
    V_n = V_at_lat_bnds.values[:, 1:]

    # get the areas of the various grid cell faces
    A_w = A_e = (
        dlat_to_meters(np.diff(lat_bnds)).reshape((1, -1, 1))
        * np.diff(z_bnds).reshape((-1, 1, 1))
    )
    dlon_at_lat_bnds = dlon_to_meters(np.diff(lon_bnds).reshape((1, 1, -1)), lat_bnds.reshape((1, -1, 1)))
    A_t = (
        dlat_to_meters(np.diff(lat_bnds).reshape((1, -1, 1)))
        * ((dlon_at_lat_bnds[:, :-1] + dlon_at_lat_bnds[:, 1:]) / 2)  # area as if grid cells are trapezoids, decent approximation
    )

    A_s = dlon_at_lat_bnds[:, :-1] * np.diff(z_bnds).reshape((-1, 1, 1))
    A_n = dlon_at_lat_bnds[:, 1:] * np.diff(z_bnds).reshape((-1, 1, 1))

    rho_profile = xr.open_dataset('../examples/configfiles/config.nc')['seawater_density']  # this should get hard-coded in better
    rho_z_bnds = rho_profile.interp(z_sd=z_bnds).values.reshape((-1, 1, 1))
    rho_c = rho_profile.interp(z_sd=UV.depth).values.reshape((-1, 1, 1))

    U_w[np.isnan(U_w)] = 0  # treat all non-ocean cells as just 0 velocity, fine since we're summing things
    U_e[np.isnan(U_e)] = 0
    V_s[np.isnan(V_s)] = 0
    V_n[np.isnan(V_n)] = 0
    horizontal_mass_flux = (
        rho_c * ((A_w*U_w - A_e*U_e) + (A_s*V_s - A_n*V_n))
    )
    # now mask such that we constrain domain to true ocean cells
    horizontal_mass_flux[np.isnan(UV.U.values)] = np.nan

    # Step 3
    if verbose:
        print("Vertically integrating mass flux")
    vertical_mass_flux = np.concatenate([
        np.zeros((1, len(UV.lat), len(UV.lon))),
        np.nancumsum(horizontal_mass_flux, axis=0)
    ], axis=0)
    # again, mask this such that we constrain domain to tops/bottoms of ocean cells
    vertical_mass_flux[:-1][np.isnan(UV.U.values)] = np.nan  # [:-1] to map cell centers to cell bottoms
    vertical_mass_flux[-1][np.isnan(UV.U.values[-1])] = np.nan  # to map top level centers to cell tops

    w_trad = vertical_mass_flux / (rho_z_bnds * A_t)

    if verbose:
        print("Adjusting traditional profile based on boundary conditions...")
    # aka calculate adjoint profile with no weight on continuity
    w_surface = 0
    # generate a bathymetry matrix:
    h = z_bnds[np.argmax(~np.isnan(UV.U.values), axis=0)]  # depth of the first non-nan grid cell, going from bottom up
    h[np.isnan(UV.U.values[-1])] = 0   # of course, if whole column is nans (land), then the argmax returns 0.  fix this

    w_c = (w_surface - w_trad[-1]) * (h - z_bnds.reshape(-1, 1, 1))/h  # linear correction of profile enforcing w(z=0) = 0
    w_adj = w_trad + w_c

    # interpolate the result back onto the original z grid points
    W_calc = xr.DataArray(data=w_adj, dims=('depth', 'lat', 'lon'),
                          coords={'depth': z_bnds, 'lat': UV.lat, 'lon': UV.lon})
    W_calc = W_calc.interp(depth=UV.depth).sortby('depth', ascending=False)
    return W_calc


def dlat_to_meters(dlat: np.ndarray):
    """
    :param dlat: displacement in latitude (degrees N)
    :return: length of displacement in meters
    """
    return dlat*111111


def dlon_to_meters(dlon: np.ndarray, lat: np.ndarray):
    """
    :param dlon: displacement in longitude (degrees E)
    :param lat: latitude of displacement (degrees N)
    :return: length of displacement in meters, assuming spherical earth
    """
    return dlon * 111111 * np.cos(np.deg2rad(lat))


def compare_Ws(W1: Tuple, W2: Tuple, level: int, clip=1e-4):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, (name, W) in zip(axes, (W1, W2)):
        cf = ax.contourf(W.lon, W.lat, W.isel(depth=level).clip(-clip, clip), cmap='bwr', levels=30, vmin=-clip, vmax=clip)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.ax.set_ylabel('W (m/s)')
        ax.set_title(f"{name} (z = {float(W.depth.isel(depth=level)): .0f} m)")
    plt.tight_layout()
