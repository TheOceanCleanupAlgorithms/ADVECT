import xarray as xr

SOURCEFILE_VARIABLES = ['lon', 'lat', 'releaseDate', 'unsd']


def open_sourcefile(sourcefile_path: str) -> xr.Dataset:
    sourcefile = xr.open_dataset(sourcefile_path)

    # make sure there's only one dimension, and replace it with 'id' if it isn't already.
    dims = list(sourcefile.dims.keys())
    assert len(dims) == 1, "sourcefile has more than one dimension."
    if dims[0] != 'id':
        assert 'id' in sourcefile.data_vars, "sourcefile does not include 'id' as a dimension or variable."
        sourcefile = sourcefile.swap_dims({dims[0]: 'id'})

    for var in SOURCEFILE_VARIABLES:
        assert var in sourcefile.data_vars, f"sourcefile does not include variable '{var}'"

    return sourcefile
