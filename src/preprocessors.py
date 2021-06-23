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
