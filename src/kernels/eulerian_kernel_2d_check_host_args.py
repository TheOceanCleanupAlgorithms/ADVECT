"""
Since we can't raise errors inside kernels, the best practice is to create an arg-checking function which accompanies
every kernel.  Before calling any kernel, its arg-check function should be called.
"""
import opencl_specification_constants as cl_const
import numpy as np


def check_args(host_bufs):
    field_x = host_bufs['h_field_x']
    assert max(field_x) < 180
    assert min(field_x) >= -180
    assert len(field_x) <= cl_const.UINT_MAX + 1
    assert is_uniformly_spaced(field_x)

    field_y = host_bufs['h_field_y']
    assert max(field_y) < 90
    assert min(field_y) >= -90
    assert len(field_y) <= cl_const.UINT_MAX + 1
    assert is_uniformly_spaced(field_y)

    field_t = host_bufs['h_field_t']
    assert len(field_t) <= cl_const.UINT_MAX + 1
    assert is_uniformly_spaced(field_t)

    x0 = host_bufs['h_x0']
    assert max(x0) < 180
    assert min(x0) >= -180

    y0 = host_bufs['h_y0']
    assert max(y0) < 90
    assert min(y0) >= -90


def is_uniformly_spaced(arr):
    tol = 1e-5
    return all(np.abs(np.diff(arr) - np.diff(arr)[0]) < tol)
