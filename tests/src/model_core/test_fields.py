from pathlib import Path
from typing import List

import numpy as np
import pyopencl as cl

from tests.config import CL_CONTEXT, CL_QUEUE, MODEL_CORE_DIR

KERNEL_SOURCE = Path(__file__).with_suffix('.cl')
CL_PROGRAM = cl.Program(CL_CONTEXT, open(KERNEL_SOURCE).read()).build(
    options=["-I", str(MODEL_CORE_DIR)])


def x_is_circular(x: np.ndarray) -> np.ndarray:
    """
    :param x: a sorted, ascending, equally spaced array representing longitude in domain [-180, 180]
    """
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "fields.cl"
    #include "geography.cl"

    __kernel void test_x_is_circular(
        __global const double* x,
        const unsigned int x_len,
        __global unsigned int *out) {

        field3d field = {.x = x, .x_len = x_len, .x_spacing = calculate_spacing(x, x_len)};
        out[0] = x_is_circular(field);
    }
    """).build(options=["-I", str(MODEL_CORE_DIR)])

    d_x = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x.astype(np.float64))

    out = np.zeros(1).astype(np.uint32)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_x_is_circular(CL_QUEUE, (1,), None, d_x, np.uint32(len(x)), d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_x_is_circular():
    assert x_is_circular(np.arange(-180, 180))
    assert not x_is_circular(np.linspace(10, 40, 50))
    assert not x_is_circular(np.arange(-181, 181))
    assert not x_is_circular(np.array([0]))


def field_element_is_null(gp: dict, field: dict) -> np.ndarray:
    """
    :param gp: grid point, keys {'x_idx', 'y_idx', 'z_idx', 't_idx}, values are ints
    :param field: 3d vector field, keys {'x', 'y', 'z', 't', 'U', 'V', 'W'}, values are ndarrays
    """
    # setup
    d_field = create_field_buffers(field)

    out = np.zeros(1).astype(np.uint32)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    CL_PROGRAM.test_field_element_is_null(
        CL_QUEUE, (1,), None,
        d_field['x'], np.uint32(len(field['x'])),
        d_field['y'], np.uint32(len(field['y'])),
        d_field['z'], np.uint32(len(field['z'])),
        d_field['t'], np.uint32(len(field['t'])),
        d_field['U'], np.uint32('U' in field.keys()),
        d_field['V'], np.uint32('V' in field.keys()),
        d_field['W'], np.uint32('W' in field.keys()),
        np.uint32(gp['x_idx']),
        np.uint32(gp['y_idx']),
        np.uint32(gp['z_idx']),
        np.uint32(gp['t_idx']),
        d_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out[0]


def create_field_buffers(field: dict) -> dict:
    d_field = {}
    for key in {'x', 'y', 'z', 't'}:
        d_field[key] = cl.Buffer(
                CL_CONTEXT,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=np.float64(field[key])
        )
    for key in {'U', 'V', 'W'}:
        if key in field.keys():
            key_values = field[key]
        else:
            key_values = np.empty((len(field['t']), len(field['z']), len(field['y']), len(field['x'])))
        d_field[key] = cl.Buffer(
                CL_CONTEXT,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=np.float32(key_values).flatten()
        )
    return d_field


def test_field_element_is_null():
    gp = {'x_idx': 0, 'y_idx': 0, 'z_idx': 0, 't_idx': 0}
    base_field = {
        'x': np.zeros(1), 'y': np.zeros(1), 'z': np.zeros(1), 't': np.zeros(1),
        'U': np.zeros(1), 'V': np.zeros(1), 'W': np.zeros(1),
    }

    # test base field element is non-null
    assert not field_element_is_null(gp, base_field)

    # test field element with a null in one component is considered null
    U_null = dict(base_field, U=np.full(1, np.nan))
    assert field_element_is_null(gp, U_null)
    V_null = dict(base_field, V=np.full(1, np.nan))
    assert field_element_is_null(gp, V_null)
    W_null = dict(base_field, W=np.full(1, np.nan))
    assert field_element_is_null(gp, W_null)

    # test field element with a null in two components is also considered null
    UV_null = dict(U_null, V=np.full(1, np.nan))
    assert field_element_is_null(gp, UV_null)

    # test field element with a null in all three components is also considered null
    UVW_null = dict(UV_null, W=np.full(1, np.nan))
    assert field_element_is_null(gp, UVW_null)

    # test a non-null field element with an undefined component is still valid
    W_undefined_valid = base_field.copy()
    del W_undefined_valid['W']
    assert not field_element_is_null(gp, W_undefined_valid)

    # test a null field element with an undefined component is still invalid
    W_undefined_invalid = V_null.copy()
    del W_undefined_invalid['W']
    assert field_element_is_null(gp, W_undefined_invalid)

    # test a non-null field element with TWO undefined components is still valid
    VW_undefined = W_undefined_valid.copy()
    del VW_undefined['V']
    assert not field_element_is_null(gp, VW_undefined)

    # vacuously, a field element with THREE undefined components is still valid
    UVW_undefined = VW_undefined.copy()
    del UVW_undefined['U']
    assert not field_element_is_null(gp, UVW_undefined)


def double_jack_search(gp: dict, field: dict, modular_x: bool = False) -> np.ndarray:
    """
    :param gp: grid point, keys {'x_idx', 'y_idx', 'z_idx', 't_idx}, values are ints
    :param field: 3d vector field, keys {'x', 'y', 'z', 't', 'U', 'V', 'W'}, values are ndarrays
    :param modular_x: whether the x coordinate of "field" wraps around (e.g. longitude)
    """
    d_field = create_field_buffers(field)
    out = np.zeros(3).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    CL_PROGRAM.test_double_jack_search(
        CL_QUEUE, (1,), None,
        d_field['x'], np.uint32(len(field['x'])),
        d_field['y'], np.uint32(len(field['y'])),
        d_field['z'], np.uint32(len(field['z'])),
        d_field['t'], np.uint32(len(field['t'])),
        d_field['U'],
        np.uint32(modular_x),
        np.uint32(gp['x_idx']),
        np.uint32(gp['y_idx']),
        np.uint32(gp['z_idx']),
        np.uint32(gp['t_idx']),
        d_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_double_jack_search():
    zero_gp = {'x_idx': 0, 'y_idx': 0, 'z_idx': 0, 't_idx': 0}

    # test singleton field with valid element at origin
    valid_singleton_field = {
        'x': np.zeros(1), 'y': np.zeros(1), 'z': np.zeros(1), 't': np.zeros(1),
        'U': np.zeros(1),
    }
    np.testing.assert_array_equal(
        double_jack_search(gp=zero_gp, field=valid_singleton_field),
        [0, np.nan, np.nan],
    )

    # test singleton field with invalid element at origin
    invalid_singleton_field = dict(valid_singleton_field, U=np.full(1, np.nan))
    np.testing.assert_array_equal(
        double_jack_search(gp=zero_gp, field=invalid_singleton_field),
        [np.nan, np.nan, np.nan],
    )

    def seek_a_point(origin: tuple, valids: List[tuple], modular_x: bool):
        """
        creates a field full of nans, sets specified elements to specified values, then
            calls a search on the field starting at origin.  Returns found value.
        :param origin: tuple (x_idx, y_idx)
        :param valids: List of tuples (x_idx, y_idx, value)
        :param modular_x: whether or not
        x indices must be integers 0 to 10
        y incices: must be integers 0 to 4
        """
        origin = dict(x_idx=origin[0], y_idx=origin[1], z_idx=0, t_idx=0)
        empty_field = {
            'x': np.linspace(-10, 10, 11), 'y': np.linspace(-4, 4, 5),
            'z': np.zeros(1), 't': np.zeros(1),
            'U': np.full((5, 11), np.nan)
        }
        for x, y, value in valids:
            empty_field['U'][y, x] = value
        return double_jack_search(gp=origin, field=empty_field, modular_x=modular_x)[0]

    for modular_x in (True, False):
        # check one north
        assert not np.isnan(seek_a_point(origin=(4, 2), valids=[(4, 3, 0)], modular_x=modular_x))
        # check one east
        assert not np.isnan(seek_a_point(origin=(4, 2), valids=[(5, 2, 0)], modular_x=modular_x))
        # check one southwest
        assert not np.isnan(seek_a_point(origin=(4, 2), valids=[(3, 1, 0)], modular_x=modular_x))
        # check one far away
        assert not np.isnan(seek_a_point(origin=(0, 0), valids=[(10, 0, 0)], modular_x=modular_x))
        # check one that should be missed by search pattern
        assert np.isnan(seek_a_point(origin=(8, 4), valids=[(6, 3, 0)], modular_x=modular_x))
        # check that cross is chosen over corner
        assert seek_a_point(origin=(5, 2), valids=[(6, 2, 0), (6, 3, 1)], modular_x=modular_x) == 0
        # check that radius 1 is chosen over radius 2
        assert seek_a_point(origin=(5, 2), valids=[(6, 3, 0), (5, 4, 1)], modular_x=modular_x) == 0
    # test modular x axis; point across modular divide should be closer
    assert seek_a_point(origin=(0, 2), valids=[(3, 2, 0), (10, 2, 1)], modular_x=True) == 1

    # check one above
    depth_field = {
        'x': np.linspace(-10, 10, 11), 'y': np.linspace(-4, 4, 5),
        'z': np.linspace(-2, 2, 5), 't': np.zeros(1),
        'U': np.full((5, 5, 11), np.nan)
    }
    origin = dict(x_idx=5, y_idx=4, z_idx=1, t_idx=0)
    depth_field["U"][2, 4, 5] = 0  # find point above
    assert double_jack_search(gp=origin, field=depth_field)[0] == 0
    depth_field["U"][0, 4, 5] = 1  # find point below before point above
    assert double_jack_search(gp=origin, field=depth_field)[0] == 1
