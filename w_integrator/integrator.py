"""
Step 1: calculate horizontal velocities at the boundaries between cells (for whole dataset!)
Step 2: do a 2x2 convolution on this to get the horizontal mass flux into every cell
Step 3: vertically integrate
"""
from typing import Tuple

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar


rho_profile = xr.open_dataset("../examples/configfiles/config.nc")["seawater_density"]
# this should get hard-coded in better


def generate_vertical_velocity(
    UV: xr.Dataset,
    verbose: bool = False,
):
    """
    :param UV: xarray Dataset containing 3d zonal and meridional current
        expected dimensions: {'depth' (ascending), 'lat' (uniform spacing), 'lon' (uniform spacing)}
        expected variables: {'U', 'V'}
    :param verbose: extra printing
    :return:
    """

    np.testing.assert_allclose(np.diff(UV.lon), np.mean(np.diff(UV.lon)), rtol=0.001)
    np.testing.assert_allclose(np.diff(UV.lat), np.mean(np.diff(UV.lat)), rtol=0.001)
    assert np.all(np.diff(UV.depth) > 0)

    UV = UV.chunk({"depth": 1})  # convert to dask arrays chunked by depth levels
    # Step 1
    if verbose:
        print("calculating velocities at the boundaries between cells")
    # first, generate new grids for the edges of cells

    lat_bnds, lon_bnds = [
        np.concatenate(
            [
                coord[:1] - np.diff(coord)[0] / 2,
                coord[:-1] + np.diff(coord) / 2,
                coord[-1:] + np.diff(coord)[-1] / 2,
            ]
        )
        for coord in (UV.lat.values, UV.lon.values)
    ]  # first and last bnd are the same point for lon_bnds

    z_bnds = np.zeros(len(UV.depth) + 1)
    for i in reversed(range(len(z_bnds) - 1)):
        z_bnds[i] = float(2 * UV.depth[i] - z_bnds[i + 1])

    # coordinates of V_at_lat_bnds: (depth, lat_bnds, lon)
    V_at_lat_bnds = UV.V.interp(lat=lat_bnds, kwargs={"fill_value": 0}).rename(
        {"lat": "lat_bnds"}
    )
    # fill_value of zero applies to extrapolation, aka the first and last lat_bound.
    # The assumption is that flow is zero at the outermost latitude bounds.

    U_at_lon_bnds = UV.U.interp(
        lon=lon_bnds[1:-1],
    ).rename({"lon": "lon_bnds"})
    # the first/last lat_bnd, which are outside the range of interpolation,
    # are set to the linear interpolation across the meridian where longitude modularly wraps
    modular_meridian = (
        UV.U.isel({"lon": [0, -1]})
        .mean(dim="lon", keepdims=True)
        .rename({"lon": "lon_bnds"})
    )
    U_at_lon_bnds = xr.concat(
        (
            modular_meridian.assign_coords({"lon_bnds": lon_bnds[:1]}),
            U_at_lon_bnds,
            modular_meridian.assign_coords({"lon_bnds": lon_bnds[-1:]}),
        ),
        dim="lon_bnds",
    )

    # Step 2
    if verbose:
        print("Calculating mass flux into every cell due to U/V...")
    # we offset the bounds arrays a bit so that we wind up with things the same shape as the cell grid; each value
    # thus corresponds to the flux across the west (e.g.) boundary of a particular grid cell
    ds = xr.Dataset(coords=UV.coords).chunk({"depth": 1})
    ds["U_west"] = (
        U_at_lon_bnds.isel(lon_bnds=slice(None, -1))
        .rename({"lon_bnds": "lon"})
        .assign_coords({"lon": UV.lon})
    )
    ds["U_east"] = (
        U_at_lon_bnds.isel(lon_bnds=slice(1, None))
        .rename({"lon_bnds": "lon"})
        .assign_coords({"lon": UV.lon})
    )
    ds["V_south"] = (
        V_at_lat_bnds.isel(lat_bnds=slice(None, -1))
        .rename({"lat_bnds": "lat"})
        .assign_coords({"lat": UV.lat})
    )
    ds["V_north"] = (
        V_at_lat_bnds.isel(lat_bnds=slice(1, None))
        .rename({"lat_bnds": "lat"})
        .assign_coords({"lat": UV.lat})
    )

    # treat all non-ocean cells as just 0 velocity, fine since we're summing things
    ds = ds.fillna(0)

    # get the areas of the various grid cell faces
    # assuming lat/lon are equally spaced arrays
    dlat = dlat_to_meters(np.diff(lat_bnds).mean())  # meters
    dlon_at_lat_bnds = dlon_to_meters(np.diff(lon_bnds).mean(), lat_bnds)  # meters
    ds["A_eastwest"] = (
        "depth",
        (dlat * np.diff(z_bnds)).astype(np.float32),
    )
    ds["A_topbottom"] = (
        "lat",
        (dlat * ((dlon_at_lat_bnds[:-1] + dlon_at_lat_bnds[1:]) / 2)).astype(np.float32),
    )  # area as if grid cells are trapezoids, decent approximation
    ds["A_south"] = (
        ("depth", "lat"),
        (dlon_at_lat_bnds[:-1].reshape((1, -1)) * np.diff(z_bnds).reshape((-1, 1))).astype(np.float32),
    )
    ds["A_north"] = (
        ("depth", "lat"),
        (dlon_at_lat_bnds[1:].reshape((1, -1)) * np.diff(z_bnds).reshape((-1, 1))).astype(np.float32),
    )
    ds["rho"] = ("depth", (rho_profile.interp(z_sd=UV.depth).values).astype(np.float32))

    # do ze calculation!
    horizontal_mass_flux = ds["rho"] * (
        (ds["A_eastwest"] * (ds["U_west"] - ds["U_east"]))
        + (ds["A_south"] * ds["V_south"] - ds["A_north"] * ds["V_north"])
    )
    # now mask such that we constrain domain to true ocean cells
    horizontal_mass_flux = horizontal_mass_flux.where(~UV.U.isnull())

    # Step 3
    if verbose:
        print("Vertically integrating mass flux")
    vertical_mass_flux = (
        horizontal_mass_flux.cumsum(dim="depth", skipna=True)
        .rename({"depth": "depth_bnds"})
        .assign_coords({"depth_bnds": z_bnds[1:]})
    )

    # set mass flux of 0 at grid bottom
    vertical_mass_flux = xr.concat(
        (
            xr.zeros_like(
                vertical_mass_flux.isel(depth_bnds=0).assign_coords(
                    {"depth_bnds": z_bnds[0]}
                ),
            ),
            vertical_mass_flux,
        ),
        dim="depth_bnds",
    )

    # mask such that flux is only valid if either cell above or below the flux across cell boundary is an ocean cell
    vertical_mass_flux = vertical_mass_flux.where(
        ~xr.concat((UV.U.isel(depth=0), UV.U), dim="depth")
        .isnull()
        .rename({"depth": "depth_bnds"})
        .assign_coords({"depth_bnds": z_bnds})
    )

    rho_z_bnds = rho_profile.interp(z_sd=z_bnds).rename({"z_sd": "depth_bnds"}).astype(np.float32)
    w_trad = vertical_mass_flux / (rho_z_bnds * (ds["A_topbottom"]))

    if verbose:
        print("Adjusting traditional-method profile based on boundary conditions...")
    # aka calculate adjoint profile with no weight on continuity
    # get the ocean depth at the bottom of every lat/lon vertical column by looking for the first non-null element in
    # each column.  This doesn't handle columns on continents correctly (as they are entirely null elements);
    # the .where takes care of these cells.
    h = (
        (~w_trad.isnull())
        .idxmax(dim="depth_bnds")
        .where(~w_trad.isel(depth_bnds=-1).isnull(), 0)
    ).astype(np.float32)

    true_w_surface = 0  # the ocean-air interface boundary condition!
    # so-called adjoint method: apply a correction to the profile such that the top boundary condition is satisfied;
    # this correction decreases linearly with depth, such that the correction at the column floor is 0, thus preserving
    # the bottom boundary condition as well.
    w_c = (
        (true_w_surface - w_trad.isel(depth_bnds=-1)) * (h - w_trad["depth_bnds"].astype(np.float32)) / h
    )  # linear correction of profile enforcing w(z=0) = 0
    w_adj = w_trad + w_c

    W_orig_grid = w_adj.interp(depth_bnds=UV.depth.values).rename(
        {"depth_bnds": "depth"}
    )

    W_orig_grid.name = "W"
    # should add attributes here too...
    return W_orig_grid


def dlat_to_meters(dlat: np.ndarray):
    """
    :param dlat: displacement in latitude (degrees N)
    :return: length of displacement in meters
    """
    return dlat * 111111


def dlon_to_meters(dlon: np.ndarray, lat: np.ndarray):
    """
    :param dlon: displacement in longitude (degrees E)
    :param lat: latitude of displacement (degrees N)
    :return: length of displacement in meters, assuming spherical earth
    """
    return dlon * 111111 * np.cos(np.deg2rad(lat))


def compare_Ws(W1: Tuple, W2: Tuple, depth: float, clip=1e-4):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, (name, W) in zip(axes, (W1, W2)):
        cf = ax.contourf(
            W.lon,
            W.lat,
            W.sel(depth=depth, method='nearest').clip(-clip, clip),
            cmap="bwr",
            levels=30,
            vmin=-clip,
            vmax=clip,
        )
        cbar = plt.colorbar(cf, ax=ax)
        cbar.ax.set_ylabel("W (m/s)")
        ax.set_title(f"{name} (z = {float(W.depth.sel(depth=depth, method='nearest')): .0f} m)")
    plt.tight_layout()
