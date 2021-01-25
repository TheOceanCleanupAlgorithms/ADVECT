from datetime import datetime
from datetime import timedelta
import glob
from typing import Optional

import xarray as xr
import pandas as pd

SOURCEFILE_VARIABLES = {'p_id', 'lon', 'lat', 'depth', 'radius', 'density', 'corey_shape_factor', 'release_date'}


def open_sourcefiles(
    sourcefile_path: str,
    variable_mapping: Optional[dict],
) -> xr.Dataset:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param variable_mapping: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('p_id', 'lat', 'lon', 'depth', 'release_date')
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

    for var in SOURCEFILE_VARIABLES:
        assert var in sourcefile.variables, f"missing variable '{var}'.  If differently named, pass in mapping."

    sourcefile.load()  # persist into RAM

    return sourcefile
