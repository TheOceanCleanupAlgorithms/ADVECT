from collections import OrderedDict
from typing import List

import numpy as np
import pyopencl as cl
import xarray as xr

from kernel_wrappers import kernel_constants


def is_uniformly_spaced_ascending(arr):
    tol = 1e-3
    return len(arr) == 1 or all(np.abs(np.diff(arr) - np.diff(arr)[0]) < tol)


def is_sorted_ascending(arr):
    return np.all(np.diff(arr) > 0)


def buffer_from_array(arr: np.ndarray, context: cl.Context) -> cl.Buffer:
    return cl.Buffer(
        context,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=arr.ravel(),
    )


class Field3D:
    """
    python wrapping of field3d, from src/model_core/fields.h.
    Enforces correct datatypes, provides convenience functions for kernel argument creation, etc.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        varnames: List[str],
        non_uniform_time: bool = False,
    ):
        """
        :param ds: xarray dataset with some variables with coordinates {"time", "depth" (optional), "lat", "lon"},
        :param varnames: ordered list of variable names in ds (max 4 [U, V, W, bathy])
        """
        if len(varnames) < 1 or len(varnames) > 4:
            raise ValueError("Field3D must have between 1-4 variables")

        # create coordinate arrays, correct datatype, enforced spacing/sorting requirements, persisted into RAM
        self.coords = OrderedDict()
        self.coords["x"] = ds.lon.values.astype(np.float64)
        assert max(self.coords["x"]) <= 180
        assert min(self.coords["x"]) >= -180
        assert 1 <= len(self.coords["x"]) <= kernel_constants.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.coords["x"])
        self.coords["y"] = ds.lat.values.astype(np.float64)
        assert max(self.coords["y"]) <= 90
        assert min(self.coords["y"]) >= -90
        assert 1 <= len(self.coords["y"]) <= kernel_constants.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.coords["y"])
        if "depth" in ds.dims:
            self.coords["z"] = ds.depth.values.astype(np.float64)
            assert max(self.coords["z"]) <= 0
            assert is_sorted_ascending(self.coords["z"])
        # float64 representation of unix timestamp
        self.coords["t"] = ds.time.values.astype("datetime64[s]").astype(np.float64)
        assert 1 <= len(self.coords["t"]) <= kernel_constants.UINT_MAX + 1
        if non_uniform_time:
            assert is_sorted_ascending(self.coords["t"])
        else:
            assert is_uniformly_spaced_ascending(self.coords["t"])

        # put variables in an ordered dict, correct datatype, persisted into RAM
        if "z" in self.coords:
            ds = ds.transpose("time", "depth", "lat", "lon")
        else:
            ds = ds.transpose("time", "lat", "lon")
        self.variables = OrderedDict(
            (
                var,
                ds[var].astype(np.float32).values,
            )
            for var in varnames
        )

    def create_kernel_arguments(self, context: cl.Context) -> list:
        args = []
        for coord in self.coords.values():
            args.append(buffer_from_array(coord, context))
            args.append(np.uint32(len(coord)))
        for var in self.variables.values():
            args.append(buffer_from_array(var, context))
        return args

    def memory_usage_bytes(self) -> int:
        return sum(coord.nbytes for coord in self.coords.values()) + sum(
            var.nbytes for var in self.variables.values()
        )


def create_empty_2d_field():
    dummy_var = (["time", "lat", "lon"], [[[0]]])
    return Field3D(
        ds=xr.Dataset(
            data_vars={"U": dummy_var, "V": dummy_var},
            coords={"time": [0], "lat": [0], "lon": [0]},
        ),
        varnames=["U", "V"],
    )
