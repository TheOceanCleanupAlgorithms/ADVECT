import numpy as np
import xarray as xr
from ecco_v4_py import resample_to_latlon
from tqdm import tqdm
import glob
from pathlib import Path


def interpolate_variable(
    ECCO_varname,
    local_varname,
    new_grid_min_lat,
    new_grid_max_lat,
    new_grid_delta_lat,
    new_grid_min_lon,
    new_grid_max_lon,
    new_grid_delta_lon,
    native_grid_dir: Path,
    interp_grid_dir: Path,
):
    print(f"Interpolating all {ECCO_varname} files...")
    ECCO_grid = xr.open_dataset(native_grid_dir / "ECCO-GRID.nc")
    files = sorted(glob.glob(str(native_grid_dir / f"{ECCO_varname}*.nc")))
    for file in tqdm(files):
        ds = xr.open_dataset(file)
        date = ds.time.dt.strftime("%Y-%m-%d").values[0]
        out_path = interp_grid_dir / f"{local_varname}_{date}.nc"
        if out_path.exists():
            continue

        if ECCO_varname == "WVELMASS":
            # interpolate vertical grid to Z from Zl; change vertical indexer name
            ds[ECCO_varname] = (
                ds[ECCO_varname]
                .swap_dims({"k_l": "Zl"})
                .interp(Zl=ECCO_grid.Z.values)
                .rename({"Zl": "Z", "k_l": "k"})
                .swap_dims({"Z": "k"})
                .assign_coords({"k": ECCO_grid.k.values})
            )

        interp_levels = []
        for lev in ds.k:
            new_grid_lon, new_grid_lat, var_interp = resample_to_latlon(
                ds.XC,
                ds.YC,
                ds[ECCO_varname]
                .sel(k=lev)
                .where(ECCO_grid.maskC.sel(k=lev) > 0),  # mask land to nans
                new_grid_min_lat,
                new_grid_max_lat,
                new_grid_delta_lat,
                new_grid_min_lon,
                new_grid_max_lon,
                new_grid_delta_lon,
                fill_value=np.NaN,
                mapping_method="nearest_neighbor",
            )
            interp_levels.append(var_interp)

        field_interpd_to_latlon = xr.Dataset(
            data_vars={
                local_varname: (
                    ["time", "depth", "lat", "lon"],
                    np.array(interp_levels)[np.newaxis],
                )
            },
            coords={
                "time": ds.time.values,
                "depth": ECCO_grid.Z.values,
                "lat": new_grid_lat[:, 0].astype("float64"),
                "lon": new_grid_lon[0].astype("float64"),
            },
        )

        field_interpd_to_latlon.attrs = ds.attrs
        field_interpd_to_latlon[local_varname].attrs = ds[ECCO_varname].attrs
        field_interpd_to_latlon.time.attrs = ds.time.attrs
        field_interpd_to_latlon.depth.attrs = ECCO_grid.Z.attrs
        field_interpd_to_latlon.lat.attrs = {
            "units": "degrees_north",
            "long_name": "latitude",
        }
        field_interpd_to_latlon.lon.attrs = {
            "units": "degrees_east",
            "long_name": "longitude",
        }

        field_interpd_to_latlon.to_netcdf(out_path)


def interp_ECCO_currents(
    native_grid_dir: Path, interp_grid_dir: Path, resolution_deg: float
):
    variables = {
        "EVEL": "U",  # ECCO_native varname: [local varname, vertical grid name]
        "NVEL": "V",
        "WVELMASS": "W",
    }
    for ECCO_varname, local_varname in variables.items():
        new_grid_delta_lat = resolution_deg
        new_grid_delta_lon = resolution_deg
        new_grid_min_lat = (
            -90 + new_grid_delta_lat / 2
        )  # domain of interpolated field (deg)
        new_grid_max_lat = 90 - new_grid_delta_lat / 2
        new_grid_min_lon = -180 + new_grid_delta_lon / 2
        new_grid_max_lon = 180 - new_grid_delta_lon / 2
        interpolate_variable(
            ECCO_varname,
            local_varname,
            new_grid_min_lat,
            new_grid_max_lat,
            new_grid_delta_lat,
            new_grid_min_lon,
            new_grid_max_lon,
            new_grid_delta_lon,
            native_grid_dir=native_grid_dir,
            interp_grid_dir=interp_grid_dir,
        )
