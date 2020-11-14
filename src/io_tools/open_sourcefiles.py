from datetime import datetime
from datetime import timedelta
from enum import Enum
import glob
from typing import Optional

import xarray as xr
import pandas as pd


class SourceFileType(Enum):
    """Allows to handle different type of source files."""
    trashtracker = 0
    advector = 1


SOURCEFILE_VARIABLES = ['p_id', 'lon', 'lat', 'depth', 'release_date']


def datenum_to_datetimeNS64(datenum):
    """
    Convert Matlab datenum into Python datetime64[ns]
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """

    # Matlab datenum goes down to Jan 0 of year 0, but Python datetime only goes down to Jan 1 year 1
    # On top of that, the representation of datetime64 in nanoseconds cannot go before 1678 and after 2262 approx.

    if datenum > 612513: 
        days = datenum % 1
        return datetime.fromordinal(int(datenum)) + timedelta(days=days) - timedelta(days=366)
    else:
        return datetime(1678, 1, 1, 1, 1, 1, 1)


def open_sourcefiles(
    sourcefile_path: str,
    variable_mapping: Optional[dict],
    source_file_type: SourceFileType,
) -> pd.DataFrame:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param variable_mapping: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('p_id', 'lat', 'lon', 'depth', 'release_date')
    :param source_file_type: specify what sourcefile we have.
    """
    if variable_mapping is None:
        variable_mapping = {}
    if source_file_type == SourceFileType.trashtracker:  # merge defaults with passed map, passed map wins conflicts
        default_mapping = {'releaseDate': 'release_date', 'x': 'p_id'}
        variable_mapping = dict(default_mapping, **variable_mapping)

    # Need to make sure we concat along the right dim. If there's a mapping, use it to get the name of the axis.
    # If the "p_id" coordinate is properly defined (i.e both a variable and a dimension), this shouldn't be necessary.
    if 'p_id' not in variable_mapping.values():
        concat_dim = "p_id"
    else:
        concat_dim = next(k for k, v in variable_mapping.items() if v == 'p_id')

    sourcefile = xr.open_mfdataset(
        glob.glob(sourcefile_path),
        parallel=True,
        combine="nested",
        concat_dim=concat_dim
    )
    sourcefile = sourcefile.rename({key: value for key, value in variable_mapping.items() if key in sourcefile})

    # make sure there's only one dimension
    dims = list(sourcefile.dims.keys())
    assert len(dims) == 1, "sourcefile has more than one dimension."

    if 'p_id' not in dims:
        sourcefile = sourcefile.to_dataframe()
    else:
        sourcefile = sourcefile.to_dataframe().reset_index()  # move p_id from index to column

    for var in SOURCEFILE_VARIABLES:
        assert var in sourcefile.columns, f"missing variable '{var}'.  If differently named, pass in mapping."

    if source_file_type == SourceFileType.trashtracker:
        sourcefile['release_date'] = pd.to_datetime(sourcefile['release_date'].apply(datenum_to_datetimeNS64))
        sourcefile['lon'] = ((sourcefile.lon + 180) % 360) - 180  # enforce [-180, 180] longitude

    return sourcefile
