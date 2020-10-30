import xarray as xr
import glob


def open_netcdf_vectorfield(u_path, v_path, variable_mapping):
    """
    :param u_path: wildcard path to the zonal vector files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_path: wildcard path to the meridional vector files.  See u_path for more details.
    :param variable_mapping: mapping from names in vector file to advector standard variable names
    """
    U = xr.open_mfdataset(sorted(glob.glob(u_path)), data_vars="minimal", parallel=True)
    V = xr.open_mfdataset(sorted(glob.glob(v_path)), data_vars="minimal", parallel=True)
    vectors = xr.merge((U, V))
    vectors = vectors.rename(variable_mapping)
    vectors = vectors[['U', 'V']]  # drop any additional variables
    vectors = vectors.squeeze()  # remove any singleton dimensions

    if "depth" in vectors.dims:
        vectors = vectors.sel(depth=0, method='nearest')

    assert set(vectors.dims) == {'lat', 'lon', 'time'}, f"Unexpected/missing dimension(s) ({vectors.dims})"
    return vectors
