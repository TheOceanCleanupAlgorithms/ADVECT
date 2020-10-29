from datetime import datetime
from datetime import timedelta
from enum import Enum
import glob
import xarray as xr
import pandas as pd

class SourceFileType(Enum):
    """Allows to handle different type of source files."""
    old_source_files = 0
    new_source_files = 1

SOURCEFILE_VARIABLES = ['id', 'lon', 'lat', 'release_date']

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
        return (datetime.fromordinal(int(datenum)) \
        + timedelta(days=days) \
        - timedelta(days=366))
    else:
        return datetime(1678, 1, 1, 1, 1, 1, 1)



def open_sourcefiles(
    sourcefile_path: str,
    variable_mapping: dict,
    source_file_type: SourceFileType = SourceFileType.new_source_files,
) -> pd.DataFrame:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param variable_mapping: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('id', 'lat', 'lon', 'release_date')
    """

    # Need to make sure we concat along the right dim. If there's a mapping, use it to get the name of the axis.
    try:
        concat_dim = [k for k in variable_mapping.keys() if variable_mapping[k] == "id"][0]
    except:
        concat_dim = "id"

    sourcefile = xr.open_mfdataset(
        sorted(glob.glob(sourcefile_path)),
        parallel=True,
        combine="nested",
        concat_dim=concat_dim
    )
    sourcefile = sourcefile.rename(variable_mapping)

    # make sure there's only one dimension
    dims = list(sourcefile.dims.keys())
    assert len(dims) == 1, "sourcefile has more than one dimension."

    if 'id' not in dims:
        sourcefile = sourcefile.to_dataframe()
    else:
        sourcefile = sourcefile.to_dataframe().reset_index()  # move id from index to column

    for var in SOURCEFILE_VARIABLES:
        assert var in sourcefile.columns, f"missing variable '{var}'.  If differently named, pass in mapping."
    
    if (source_file_type == SourceFileType.old_source_files):
        sourcefile['release_date'] = pd.to_datetime(sourcefile['release_date'].apply(datenum_to_datetimeNS64))
        sourcefile['lon'][sourcefile['lon'][:] >= 180] -= 360

    return sourcefile
