import pyopencl as cl
import os
import numpy as np
from config import ROOT_DIR

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def circular_segment_area(R, r) -> np.ndarray:
    """calculate area of a circular segment using kernel code
    :param R: radius of circle
    :param r: distance from center of circle to chord forming segment
    """
    # setup
    ctx = cl.create_some_context(answers=[0, 2])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, """
    #include "windage.cl"

    __kernel void area(
        const double R,
        const double r,
        __global double *out) {

        out[0] = circular_segment_area(R, r);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.area.set_scalar_arg_dtypes([np.float64, np.float64, None])
    prg.area(queue, (1,), None, np.float64(R), np.float64(r), d_out)
    queue.finish()

    cl.enqueue_copy(queue, out, d_out)

    return out


def test_circular_segment_area():
    # split in half
    np.testing.assert_allclose(circular_segment_area(R=1, r=0), np.pi*1**2 / 2)

    # tangent line (no segment)
    np.testing.assert_allclose(circular_segment_area(R=1, r=1), 0)

    # minor segment
    R, r = 10, 8
    theta = 2 * np.arccos(r / R)
    np.testing.assert_allclose(circular_segment_area(R=R, r=r),
                               .5 * R**2 * (theta - np.sin(theta)))

    # major segment
    R, r = 5, -1
    theta = 2 * np.arccos(r / R)
    np.testing.assert_allclose(circular_segment_area(R=R, r=r),
                               .5 * R**2 * (theta - np.sin(theta)))
