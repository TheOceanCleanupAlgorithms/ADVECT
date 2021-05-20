from pathlib import Path

import numpy as np
import pyopencl as cl

from tests.config import CL_CONTEXT, CL_QUEUE, MODEL_CORE_DIR

KERNEL_SOURCE = Path(__file__).parent / "test_vector.cl"


def resolve_and_sort(x, y, z) -> np.ndarray:
    """run a vector through the opencl program"""
    test = (
        cl.Program(CL_CONTEXT, open(KERNEL_SOURCE).read())
        .build(options=["-I", str(MODEL_CORE_DIR)])
        .resolve_and_sort_test
    )

    x_out = np.zeros(3)
    y_out = np.zeros(3)
    z_out = np.zeros(3)
    d_x_out = cl.Buffer(
        CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x_out
    )
    d_y_out = cl.Buffer(
        CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y_out
    )
    d_z_out = cl.Buffer(
        CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=z_out
    )

    test.set_scalar_arg_dtypes([np.float64, np.float64, np.float64, None, None, None])
    test(
        CL_QUEUE,
        (1,),
        None,
        np.float64(x),
        np.float64(y),
        np.float64(z),
        d_x_out,
        d_y_out,
        d_z_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, x_out, d_x_out)
    cl.enqueue_copy(CL_QUEUE, y_out, d_y_out)
    cl.enqueue_copy(CL_QUEUE, z_out, d_z_out)

    return np.stack((x_out, y_out, z_out)).T


def test_resolve_sort():
    #  test sorted, backward sorted, pos/neg, mixed
    result = resolve_and_sort(1, 2, 3)
    np.testing.assert_allclose(result, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    result = resolve_and_sort(3, 2, 1)
    np.testing.assert_allclose(result, [[0, 0, 1], [0, 2, 0], [3, 0, 0]])

    result = resolve_and_sort(-1, -2, -3)
    np.testing.assert_allclose(result, [[-1, 0, 0], [0, -2, 0], [0, 0, -3]])

    result = resolve_and_sort(-3, -2, -1)
    np.testing.assert_allclose(result, [[0, 0, -1], [0, -2, 0], [-3, 0, 0]])

    result = resolve_and_sort(-7.3, 5.2, -100)
    np.testing.assert_allclose(result, [[0, 5.2, 0], [-7.3, 0, 0], [0, 0, -100]])

    result = resolve_and_sort(357.6, -29.7, -203.0)
    np.testing.assert_allclose(result, [[0, -29.7, 0], [0, 0, -203.0], [357.6, 0, 0]])
