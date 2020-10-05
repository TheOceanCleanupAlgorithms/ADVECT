"""
Since we can't raise errors inside kernels, the best practice is to create an arg-checking function which accompanies
every kernel.  Before calling any kernel, its arg-check function should be called.
"""
import opencl_specification_constants as cl_const
import numpy as np


def check_args(field_x, x_len,
               field_y, y_len,
               field_t, t_len,
               field_U, field_V,
               x0, y0, t0,
               dt, ntimesteps, save_every,
               X_out, Y_out):
    assert max(field_x) < 180
    assert min(field_x) >= -180
    assert len(field_x) <= cl_const.UINT_MAX + 1
    assert is_uniformly_spaced(field_x)

    assert max(field_y) < 90
    assert min(field_y) >= -90
    assert len(field_y) <= cl_const.UINT_MAX + 1
    assert is_uniformly_spaced(field_y)

    assert len(field_t) <= cl_const.UINT_MAX + 1
    assert is_uniformly_spaced(field_t)

    assert max(x0) < 180
    assert min(x0) >= -180

    assert max(y0) < 90
    assert min(y0) >= -90


def is_uniformly_spaced(arr):
    tol = 1e-5
    return all(np.abs(np.diff(arr) - np.diff(arr)[0]) < tol)
