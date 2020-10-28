import xarray as xr
import glob


def open_currentfiles(u_path, v_path, variable_mapping):
    """
    :param u_path: wildcard path to the zonal current files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_path: wildcard path to the zonal current files.  See u_path for more details.
    :param variable_mapping: mapping from names in current file to advector standard variable names
            advector standard names: ('U', 'V', 'W', 'lat', 'lon', 'time', 'depth')"""
    U = xr.open_mfdataset(sorted(glob.glob(u_path)), data_vars="minimal", parallel=True)
    V = xr.open_mfdataset(sorted(glob.glob(v_path)), data_vars="minimal", parallel=True)
    currents = xr.merge((U, V))
    currents = currents.rename(variable_mapping)

    if "depth" in currents.dims:
        currents = currents.isel(depth=0)

    return currents
