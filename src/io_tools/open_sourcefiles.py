import glob
from typing import Optional, Set

import xarray as xr


def open_3d_sourcefiles(
    sourcefile_path: str,
    variable_mapping: Optional[dict],
) -> xr.Dataset:
    return open_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=variable_mapping,
        expected_vars={'p_id', 'lon', 'lat', 'depth', 'radius', 'density', 'corey_shape_factor', 'release_date'},
    )


def open_2d_sourcefiles(
    sourcefile_path: str,
    variable_mapping: Optional[dict],
) -> xr.Dataset:
    return open_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=variable_mapping,
        expected_vars={'p_id', 'lon', 'lat', 'release_date'},
    )


def open_sourcefiles(
    sourcefile_path: str,
    variable_mapping: Optional[dict],
    expected_vars: Set[str],
) -> xr.Dataset:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param variable_mapping: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('p_id', 'lat', 'lon', 'depth', 'release_date')
    :param expected_vars: variable names which are expected to be in sourcefile
    """
    if variable_mapping is None:
        variable_mapping = {}

    # Need to make sure we concat along the right dim. If there's a mapping, use it to get the name of the axis.
    # If the "p_id" coordinate is properly defined (i.e both a variable and a dimension), this shouldn't be necessary.
    if 'p_id' not in variable_mapping.values():
        concat_dim = "p_id"
    else:
        concat_dim = next(k for k, v in variable_mapping.items() if v == 'p_id')

    sourcefile = xr.open_mfdataset(
        sorted(glob.glob(sourcefile_path)),
        parallel=True,
        combine="nested",
        concat_dim=concat_dim
    )
    sourcefile = sourcefile.rename({key: value for key, value in variable_mapping.items() if key in sourcefile})

    # make sure there's only one dimension
    dims = list(sourcefile.dims.keys())
    assert len(dims) == 1, "sourcefile has more than one dimension."

    for var in expected_vars:
        assert var in sourcefile.variables, f"missing variable '{var}'.  If differently named, pass in mapping."

    sourcefile.load()  # persist into RAM

    return sourcefile
