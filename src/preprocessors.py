from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from typing import Dict


def desampling_preprocessor(vectors: xr.Dataset):
    tol = 0.001
    desampling_resolution = 0.32

    vectors = vectors.where(
        np.abs(
            (vectors.lon - vectors.lon[0]) / desampling_resolution
            - np.round((vectors.lon - vectors.lon[0]) / desampling_resolution)
        )
        < tol,
        drop=True,
    )
    vectors = vectors.where(
        np.abs(
            (vectors.lat - vectors.lat[0]) / desampling_resolution
            - np.round((vectors.lat - vectors.lat[0]) / desampling_resolution)
        )
        < tol,
        drop=True,
    )

    return vectors


def multiply_preprocessor(mult_factor: float):
    def func(vectors: xr.Dataset):
        vectors["U"] = vectors.U.pipe(lambda arr: arr * mult_factor)
        vectors["U"] = vectors.U.pipe(lambda arr: arr * mult_factor)

        return vectors

    return func


def rename_var_preprocessor(variable_mapping: Dict[str, str]):
    def func(vectors: xr.Dataset):
        return vectors.rename(
            {key: value for key, value in variable_mapping.items() if key in vectors}
        )

    return func


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
        return (
            datetime.fromordinal(int(datenum))
            + timedelta(days=days)
            - timedelta(days=366)
        )
    else:
        return datetime(1678, 1, 1, 1, 1, 1, 1)


def trashtracker_sourcefile_preprocessor(sourcefile: xr.Dataset):
    sourcefile["x"] = np.arange(sourcefile.releaseDate.shape[0], dtype=np.int32)

    sourcefile = sourcefile.rename({"releaseDate": "release_date", "x": "p_id"}).drop(
        "id"
    )

    dates = np.array(
        list(map(datenum_to_datetimeNS64, sourcefile["release_date"].values))
    )

    sourcefile["release_date"] = (["p_id"], dates)

    sourcefile["lon"] = (
        (sourcefile.lon + 180) % 360
    ) - 180  # enforce [-180, 180] longitude

    return sourcefile
