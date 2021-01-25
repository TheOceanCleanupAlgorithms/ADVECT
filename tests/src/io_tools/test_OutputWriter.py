import netCDF4 as nc
import numpy as np

from io_tools.OutputWriter import copy_dataset


def test_copy_dataset():
    rng = np.random.default_rng(seed=1)
    with nc.Dataset("source.nc", mode="w", diskless=True) as source:
        source.setncattr("foo", "bar")
        source.createDimension("dim0", 100)
        var0 = source.createVariable("var0", np.float64, dimensions=("dim0"),)
        var0[:] = rng.random(100)
        var0.setncattr("what am I", "I am a variable")

        with nc.Dataset("destination.nc", mode="w", diskless=True) as destination:
            copy_dataset(source, destination)

            # global attributes
            assert source.__dict__ == destination.__dict__

            # dimension
            assert source.dimensions["dim0"].name == destination.dimensions["dim0"].name
            assert source.dimensions["dim0"].size == destination.dimensions["dim0"].size

            # variable
            assert source.variables["var0"].__dict__ == destination.variables["var0"].__dict__  # variable attributes
            assert source.variables["var0"].dimensions == destination.variables["var0"].dimensions
            np.testing.assert_array_equal(source.variables["var0"][:], destination.variables["var0"][:])
