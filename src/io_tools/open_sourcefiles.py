from datetime import datetime
from datetime import timedelta
from enum import Enum
import glob
from typing import Callable, Optional

import xarray as xr
import pandas as pd


SOURCEFILE_VARIABLES = ['p_id', 'lon', 'lat', 'release_date']


def open_sourcefiles(
    sourcefile_path: str,
    preprocessor: Optional[Callable]
) -> pd.DataFrame:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    """

    sourcefile = xr.open_mfdataset(
        sorted(glob.glob(sourcefile_path)),
        parallel=True,
        preprocess=preprocessor,
        combine="nested",
        concat_dim="p_id"
    )

    # make sure there's only one dimension
    dims = list(sourcefile.dims.keys())
    assert len(dims) == 1, "sourcefile has more than one dimension."

    if 'p_id' not in dims:
        sourcefile = sourcefile.to_dataframe()
    else:
        sourcefile = sourcefile.to_dataframe().reset_index()  # move p_id from index to column

    for var in SOURCEFILE_VARIABLES:
        assert var in sourcefile.columns, f"missing variable '{var}'.  If differently named, pass in mapping."

    return sourcefile
