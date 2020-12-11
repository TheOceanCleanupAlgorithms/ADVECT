import pyopencl as cl
import os
import numpy as np
from tests.config import ROOT_DIR

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def x_is_circular(x: np.ndarray) -> np.ndarray:
    """calculate area of a circular segment using kernel code
    :param x: a sorted, ascending, equally spaced array representing longitude in domain [-180, 180]
    """
    # setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, """
    #include "fields.cl"

    __kernel void test_x_is_circular(
        __global const double* x,
        const unsigned int x_len,
        __global bool *out) {

        field3d field = {.x = x, .x_len = x_len, .x_spacing = calculate_spacing(x, x_len)};
        out[0] = x_is_circular(field);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    d_x = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x.astype(np.float64))

    out = np.zeros(1).astype(np.bool_)
    d_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_x_is_circular(queue, (1,), None, d_x, np.uint64(len(x)), d_out)
    queue.finish()

    cl.enqueue_copy(queue, out, d_out)

    return out


def test_x_is_circular():
    assert x_is_circular(np.arange(-180, 180))
    assert not x_is_circular(np.linspace(10, 40, 50))
    assert not x_is_circular(np.arange(-181, 181))
    assert not x_is_circular(np.array([0]))
