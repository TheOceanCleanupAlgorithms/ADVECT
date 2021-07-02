import glob
from typing import Optional, Set, Callable, Optional
import xarray as xr


def open_3d_sourcefiles(
    sourcefile_path: str,
    variable_mapping: Optional[dict],
) -> xr.Dataset:
    return open_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=variable_mapping,
        expected_vars={
            "p_id",
            "lon",
            "lat",
            "depth",
            "radius",
            "density",
            "corey_shape_factor",
            "release_date",
        },
    )


def open_2d_sourcefiles(
    sourcefile_path: str,
    variable_mapping: Optional[dict],
) -> xr.Dataset:
    return open_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=variable_mapping,
        expected_vars={"p_id", "lon", "lat", "release_date"},
    )


def open_sourcefiles(
    sourcefile_path: str,
    preprocessor: Optional[Callable],
    expected_vars: Set[str],
) -> xr.Dataset:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param expected_vars: variable names which are expected to be in sourcefile
    :param preprocessor: func to call on the xarray dataset to perform operation before loading in advector, such as renaming variables.
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

    for var in expected_vars:
        assert (
            var in sourcefile.variables
        ), f"missing variable '{var}'.  If differently named, pass in mapping."

    sourcefile.load()  # persist into RAM

    return sourcefile
