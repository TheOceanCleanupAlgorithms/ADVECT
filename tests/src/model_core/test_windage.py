import os

import numpy as np
import pyopencl as cl

from tests.config import MODEL_CORE_DIR

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def calculate_windage_coeff(r, z) -> np.ndarray:
    """calculate area of a circular segment using kernel code
    :param r: radius of circle
    :param z: distance from center of circle to chord forming segment
    """
    # setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(
        ctx,
        """
    #include "windage.cl"

    __kernel void coeff(
        const double r,
        const double z,
        __global double *out) {

        out[0] = calculate_windage_coeff(r, z);
    }
    """,
    ).build(options=["-I", str(MODEL_CORE_DIR)])

    out = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.coeff(queue, (1,), None, np.float64(r), np.float64(z), d_out)
    queue.finish()

    cl.enqueue_copy(queue, out, d_out)

    return out


def test_calculate_windage_coeff():
    density_ratio = 1.17e-3
    drag_ratio = 1

    # submerged
    np.testing.assert_allclose(calculate_windage_coeff(r=1, z=-1), 0)
    np.testing.assert_allclose(calculate_windage_coeff(r=1, z=-2), 0)

    # at surface
    np.testing.assert_allclose(
        calculate_windage_coeff(r=1, z=0), np.sqrt(density_ratio * drag_ratio * 1)
    )

    # partially submerged
    r, z = 1, -0.2
    emerged_area = circular_segment_area(R=r, r=-z)
    np.testing.assert_allclose(
        calculate_windage_coeff(r=r, z=z),
        np.sqrt(
            density_ratio * drag_ratio * emerged_area / (np.pi * r ** 2 - emerged_area)
        ),
    )


def circular_segment_area(R, r) -> np.ndarray:
    """calculate area of a circular segment using kernel code
    :param R: radius of circle
    :param r: distance from center of circle to chord forming segment
    """
    # setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(
        ctx,
        """
    #include "windage.cl"

    __kernel void area(
        const double R,
        const double r,
        __global double *out) {

        out[0] = circular_segment_area(R, r);
    }
    """,
    ).build(options=["-I", str(MODEL_CORE_DIR)])

    out = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.area.set_scalar_arg_dtypes([np.float64, np.float64, None])
    prg.area(queue, (1,), None, np.float64(R), np.float64(r), d_out)
    queue.finish()

    cl.enqueue_copy(queue, out, d_out)

    return out


def test_circular_segment_area():
    # split in half
    np.testing.assert_allclose(circular_segment_area(R=1, r=0), np.pi * 1 ** 2 / 2)

    # tangent line (no segment)
    np.testing.assert_allclose(circular_segment_area(R=1, r=1), 0)

    # minor segment
    R, r = 10, 8
    theta = 2 * np.arccos(r / R)
    np.testing.assert_allclose(
        circular_segment_area(R=R, r=r), 0.5 * R ** 2 * (theta - np.sin(theta))
    )

    # major segment
    R, r = 5, -1
    theta = 2 * np.arccos(r / R)
    np.testing.assert_allclose(
        circular_segment_area(R=R, r=r), 0.5 * R ** 2 * (theta - np.sin(theta))
    )
