import pyopencl as cl
import numpy as np

from tests.config import ROOT_DIR, CL_CONTEXT, CL_QUEUE


def sample_profile(var: np.ndarray, z: np.ndarray, sample_z: np.ndarray) -> np.ndarray:
    """should return linear interpolation of profile given by var, z at z=sample_z
        samples all sample_z elements in parallel"""
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "vertical_profile.cl"

    __kernel void test_sample_profile(
        __global const double *var,
        __global const double *z,
        const unsigned int len,
        __global const double *sample_z,
        __global double *out) {
        
        vertical_profile prof = {.z = z, .var = var, .len = len};
        out[get_global_id(0)] = sample_profile(prof, sample_z[get_global_id(0)]);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(sample_z.shape).astype(np.float64)

    d_var = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=var.astype(np.float64))
    d_z = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=z.astype(np.float64))
    d_sample_z = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sample_z.astype(np.float64))
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_sample_profile(
        CL_QUEUE, sample_z.shape, None,
        d_var,
        d_z,
        np.uint32(len(z)),
        d_sample_z,
        d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_interpolation():
    """check the interpolation is same as numpy linear interp"""
    var = np.linspace(0, 1000, 10)
    z = -np.logspace(3, 0, 10)

    sample_z = np.linspace(min(z), max(z), 10000)

    np.testing.assert_allclose(np.interp(sample_z, z, var),
                               sample_profile(var, z, sample_z))


def test_outside_domain():
    """check samples outside z domain evaluate to profile endpoints"""
    var = np.array([1000, 1200, 1500, 2000])
    z = np.array([-2, 0, 1, 4])
    sample_z = np.array([-2.1, -2, 4, 4.1])
    np.testing.assert_allclose([1000, 1000, 2000, 2000],
                               sample_profile(var, z, sample_z))
