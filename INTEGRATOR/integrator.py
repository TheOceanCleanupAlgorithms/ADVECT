"""
The INTEGRATOR
Tool for integrating zonal/meridional current into vertical current using conservation of mass.
See w_integration_methodology.ipynb for an explanation of the math/methodology.
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple


def generate_vertical_velocity(
    UV: xr.Dataset,
    auto_chunk=True,
    verbose: bool = False,
) -> xr.DataArray:
    """
    :param UV: xarray Dataset containing 3d zonal and meridional current
        expected coordinates: {'depth' (ascending), 'lat' (uniform spacing), 'lon' (uniform spacing)}
        expected variables: {'U', 'V'}
        Coordinates must reference cell centers.
    :param auto_chunk: whether or not to chunk UV by depth levels.  Disable if you want to do your own chunking
    :param verbose: extra printing
    :return: xarray DataArray containing vertical current, 'W'; same coordinates as UV.
    """
    # load density profile
    rho_profile = xr.open_dataarray(
        Path(__file__).parent / "seawater_density_profile.nc"
    )

    # check assumptions about data coordinates
    np.testing.assert_allclose(
        np.diff(UV.lon), np.mean(np.diff(UV.lon)), rtol=0.001
    )  # lon uniformly spaced
    np.testing.assert_allclose(
        np.diff(UV.lat), np.mean(np.diff(UV.lat)), rtol=0.001
    )  # lat uniformly spaced
    assert np.all(np.diff(UV.depth) > 0)  # depth ascending

    # depth can't be zero, as cell center can't be zero unless top cell is infinitely thin
    assert not np.any(
        UV.depth == 0
    ), "make sure your depth coordinate references cell centers."

    if auto_chunk:
        UV = UV.chunk({"depth": 1})  # convert to dask arrays chunked by depth levels

    # ----- Step 1 ----- #
    if verbose:
        print("Step 1: Calculating mass flux into every cell due to U/V...")
    lat_bnds, lon_bnds, depth_bnds = calculate_cell_bnds(UV)
    U_at_lon_bnds = interpolate_U_to_lon_bnds(UV.U, lon_bnds)
    V_at_lat_bnds = interpolate_V_to_lat_bnds(UV.V, lat_bnds)

    m_vars = collect_variables_for_mass_flux(
        UV=UV,
        U_at_lon_bnds=U_at_lon_bnds,
        V_at_lat_bnds=V_at_lat_bnds,
        rho_profile=rho_profile,
    )
    geo = collect_grid_geometry(
        UV=UV, lat_bnds=lat_bnds, lon_bnds=lon_bnds, depth_bnds=depth_bnds
    )
    if UV.chunks:  # copy the dask coordinate chunking from UV to ds, if exists
        m_vars = m_vars.chunk(UV.chunks)

    # main equation calculating the mass flux...
    horizontal_mass_flux = m_vars["rho_cell_center"] * (
        (geo["A_eastwest"] * (m_vars["U_west"] - m_vars["U_east"]))
        + (geo["A_south"] * m_vars["V_south"] - geo["A_north"] * m_vars["V_north"])
    )
    # now mask such that we constrain domain to true ocean cells
    horizontal_mass_flux = horizontal_mass_flux.where(~UV.U.isnull())

    # ----- Step 2 ----- #
    if verbose:
        print("Vertically integrating mass flux")
    vertical_mass_flux = (
        horizontal_mass_flux.cumsum(dim="depth", skipna=True)
        .rename({"depth": "depth_bnds"})
        .assign_coords({"depth_bnds": depth_bnds[1:]})
    )
    vertical_mass_flux = restore_ocean_domain(vertical_mass_flux, UV, depth_bnds)

    # convert mass flux to vertical velocity
    rho_z_bnds = (
        rho_profile.interp(depth=depth_bnds)
        .rename({"depth": "depth_bnds"})
        .astype(np.float32)
    )
    w_trad = vertical_mass_flux / (rho_z_bnds * (geo["A_topbottom"]))

    # ----- Step 3 ----- #
    if verbose:
        print("Adjusting profile to satisfy boundary condition at surface...")
    # aka calculate adjoint profile with no weight on continuity
    w_adj = apply_adjunct_method(w_trad=w_trad, w_at_surface=0)

    # interpolate to depth gridding of UV
    W_orig_grid = w_adj.interp(depth_bnds=UV.depth.values).rename(
        {"depth_bnds": "depth"}
    )

    W_orig_grid.name = "W"
    W_orig_grid.attrs["units"] = "m/s"
    W_orig_grid.attrs["positive"] = "up"
    return W_orig_grid


def calculate_cell_bnds(UV: xr.Dataset):
    """
    :param UV: Dataset with coordinates (lat, lon, depth)
    :return: tuple (lat_bnds, lon_bnds, depth_bnds), all numpy arrays
    """
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

    depth_bnds = np.zeros(len(UV.depth) + 1)
    for i in reversed(range(len(depth_bnds) - 1)):
        depth_bnds[i] = float(2 * UV.depth[i] - depth_bnds[i + 1])

    return lat_bnds, lon_bnds, depth_bnds


def interpolate_U_to_lon_bnds(U: xr.DataArray, lon_bnds) -> xr.DataArray:
    U_at_lon_bnds = U.interp(
        lon=lon_bnds[1:-1],
    ).rename({"lon": "lon_bnds"})
    # the first/last lat_bnd, which are outside the range of interpolation,
    # are set to the linear interpolation across the meridian where longitude modularly wraps
    modular_meridian = (
        U.isel({"lon": [0, -1]})
        .mean(dim="lon", keepdims=True)
        .rename({"lon": "lon_bnds"})
    )
    return xr.concat(
        (
            modular_meridian.assign_coords({"lon_bnds": lon_bnds[:1]}),
            U_at_lon_bnds,
            modular_meridian.assign_coords({"lon_bnds": lon_bnds[-1:]}),
        ),
        dim="lon_bnds",
    )


def interpolate_V_to_lat_bnds(V: xr.DataArray, lat_bnds) -> xr.DataArray:
    # coordinates of V_at_lat_bnds: (depth, lat_bnds, lon)
    V_at_lat_bnds = V.interp(lat=lat_bnds, kwargs={"fill_value": 0}).rename(
        {"lat": "lat_bnds"}
    )
    # fill_value of zero applies to extrapolation, aka the first and last lat_bound.
    # The assumption is that flow is zero at the outermost latitude bounds.
    return V_at_lat_bnds


def collect_variables_for_mass_flux(UV, U_at_lon_bnds, V_at_lat_bnds, rho_profile):
    m_vars = xr.Dataset(coords=UV.coords)
    # we offset the bounds arrays a bit so that we wind up with things the same shape as the cell grid; each value
    # thus corresponds to the flux across the west (e.g.) boundary of a particular grid cell
    m_vars["U_west"] = (
        U_at_lon_bnds.isel(lon_bnds=slice(None, -1))
        .rename({"lon_bnds": "lon"})
        .assign_coords({"lon": UV.lon})
    )
    m_vars["U_east"] = (
        U_at_lon_bnds.isel(lon_bnds=slice(1, None))
        .rename({"lon_bnds": "lon"})
        .assign_coords({"lon": UV.lon})
    )
    m_vars["V_south"] = (
        V_at_lat_bnds.isel(lat_bnds=slice(None, -1))
        .rename({"lat_bnds": "lat"})
        .assign_coords({"lat": UV.lat})
    )
    m_vars["V_north"] = (
        V_at_lat_bnds.isel(lat_bnds=slice(1, None))
        .rename({"lat_bnds": "lat"})
        .assign_coords({"lat": UV.lat})
    )

    # treat all non-ocean cells as just 0 velocity, fine since we're summing things
    m_vars = m_vars.fillna(0)

    m_vars["rho_cell_center"] = (
        "depth",
        rho_profile.interp(depth=UV.depth).values.astype(np.float32),
    )

    return m_vars


def collect_grid_geometry(UV, lat_bnds, lon_bnds, depth_bnds):
    geometry = xr.Dataset(coords=UV.coords)
    # get the areas of the various grid cell faces
    # assuming lat/lon are equally spaced arrays
    dlat = dlat_to_meters(np.diff(lat_bnds).mean())  # meters
    dlon_at_lat_bnds = dlon_to_meters(np.diff(lon_bnds).mean(), lat_bnds)  # meters
    geometry["A_eastwest"] = (
        "depth",
        (dlat * np.diff(depth_bnds)).astype(np.float32),
    )
    geometry["A_topbottom"] = (
        "lat",
        (dlat * ((dlon_at_lat_bnds[:-1] + dlon_at_lat_bnds[1:]) / 2)).astype(
            np.float32
        ),
    )  # area as if grid cells are trapezoids, decent approximation
    geometry["A_south"] = (
        ("depth", "lat"),
        (
            dlon_at_lat_bnds[:-1].reshape((1, -1))
            * np.diff(depth_bnds).reshape((-1, 1))
        ).astype(np.float32),
    )
    geometry["A_north"] = (
        ("depth", "lat"),
        (
            dlon_at_lat_bnds[1:].reshape((1, -1)) * np.diff(depth_bnds).reshape((-1, 1))
        ).astype(np.float32),
    )

    return geometry


def restore_ocean_domain(vertical_mass_flux: xr.Dataset, UV: xr.Dataset, depth_bnds):
    """clean up the vertical mass flux domain.
    1. add a layer of zeros at the model floor
    2. set mass flux = NAN at boundaries which aren't in the ocean domain according to UV
    """
    # add a layer of 0s at the bottom of vertical mass flux, to account for the model floor
    vertical_mass_flux = xr.concat(
        (
            xr.zeros_like(
                vertical_mass_flux.isel(depth_bnds=0).assign_coords(
                    {"depth_bnds": depth_bnds[0]}
                ),
            ),
            vertical_mass_flux,
        ),
        dim="depth_bnds",
    )

    # mask such that flux is only valid if either cell above or below the flux across cell boundary is an ocean cell
    vertical_mass_flux = vertical_mass_flux.where(
        (
            ~xr.concat((UV.U.isel(depth=0), UV.U), dim="depth")
            .isnull()
            .assign_coords({"depth": depth_bnds})
            | ~xr.concat((UV.U, UV.U.isel(depth=-1)), dim="depth")
            .isnull()
            .assign_coords({"depth": depth_bnds})
        ).rename({"depth": "depth_bnds"})
    )

    return vertical_mass_flux


def apply_adjunct_method(w_trad: xr.Dataset, w_at_surface: float):
    # get the ocean depth at the bottom of every lat/lon vertical column by looking for the first non-null element in
    # each column.  The idxmax doesn't handle columns on continents correctly (as they are entirely null elements);
    # the .where takes care of these cells.
    h = (
        (~w_trad.isnull())
        .idxmax(dim="depth_bnds")
        .where(~w_trad.isel(depth_bnds=-1).isnull(), 0)
    ).astype(np.float32)

    # so-called adjoint method: apply a correction to the profile such that the top boundary condition is satisfied;
    # this correction decreases linearly with depth, such that the correction at the column floor is 0, thus preserving
    # the bottom boundary condition as well.
    w_c = (
        (w_at_surface - w_trad.isel(depth_bnds=-1))
        * (h - w_trad["depth_bnds"].astype(np.float32))
        / h
    )  # linear correction of profile enforcing w(z=0) = 0
    return w_trad + w_c


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


def compare_Ws(W1: Tuple, W2: Tuple, depth: float, clip=None):
    if clip is None:
        clip = float(5 * np.std((W1[1].sel(depth=depth, method="nearest"))))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, (name, W) in zip(axes, (W1, W2)):
        cf = ax.contourf(
            W.lon,
            W.lat,
            W.sel(depth=depth, method="nearest").clip(-clip, clip),
            cmap="bwr",
            levels=30,
            vmin=-clip,
            vmax=clip,
        )
        cbar = plt.colorbar(cf, ax=ax)
        cbar.ax.set_ylabel("W (m/s)")
        ax.set_title(
            f"{name} (z = {float(W.depth.sel(depth=depth, method='nearest')): .0f} m)"
        )
    plt.tight_layout()
