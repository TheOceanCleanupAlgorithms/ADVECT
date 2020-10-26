import xarray as xr
import pandas as pd

SOURCEFILE_VARIABLES = ['id', 'lon', 'lat', 'release_date']


def open_sourcefile(sourcefile_path: str, variable_mapping: dict) -> pd.DataFrame:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param variable_mapping: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('id', 'lat', 'lon', 'release_date')
    """
    sourcefile = xr.open_dataset(sourcefile_path)
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

    return sourcefile
