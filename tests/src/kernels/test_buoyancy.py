import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from tests.config import ROOT_DIR, CL_CONTEXT, CL_QUEUE


def buoyancy_vertical_velocity(density: np.ndarray, radius: np.ndarray, corey_shape_factor: np.ndarray) -> np.ndarray:
    """
    :param density: kg m^-3
    :param radius: m
    :return: vertical velocity due to buoyancy (m)
    """
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "buoyancy.cl"

    __kernel void test_buoyancy_vertical_velocity(
        __global const double *density,
        __global const double *radius,
        __global const double *CSF,
        __global double *out) {
        int id = get_global_id(0);
        out[id] = buoyancy_vertical_velocity(radius[id], density[id], CSF[id], 1025);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    d_density = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(density))
    d_radius = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(radius))
    d_CSF = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(corey_shape_factor))
    out = np.zeros_like(density).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_buoyancy_vertical_velocity(
            CL_QUEUE, density.shape, None,
            d_density,
            d_radius,
            d_CSF,
            d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_buoyancy_vertical_velocity(plot=False):
    """check theory against observations in Kooi 2016, figure 4"""
    nsamples = 1000
    rng = np.random.default_rng(seed=1)
    rho = np.random.uniform(930, 970, nsamples)  # plastic density not stated; this mimics a selection of LDPE and HDPE
    corey_shape_factor = rng.normal(.44, .19, nsamples)

    r_small = .5 * np.linspace(.5e-3, 1.5e-3, nsamples)  # m; "size" assumed to mean diameter
    v_small_mean = .009  # m/s
    v_small_std = .004   # m/s

    r_big = .5 * np.linspace(1.5e-3, 5e-3, nsamples)  # m; "size" assumed to mean diameter
    v_big_mean = .019  # m/s
    v_big_std = .006   # m/s

    predicted_v_small = buoyancy_vertical_velocity(rho, r_small, corey_shape_factor)
    predicted_v_big = buoyancy_vertical_velocity(rho, r_big, corey_shape_factor)

    if plot:
        plt.figure()
        plt.errorbar(np.mean(r_small), v_small_mean, v_small_std, capsize=3, color='b', label='small (observed)')
        plt.plot(r_small, predicted_v_small, '.', label='small (predicted)', color='b')
        plt.errorbar(np.mean(r_big), v_big_mean, v_big_std, capsize=3, color='g', label='big (observed)')
        plt.plot(r_big, predicted_v_big, '.', color='g', label='big (predicted)')
        plt.legend()

    # test that mean is within 1 std of obvervations
    assert v_small_mean - v_small_std < np.mean(predicted_v_small) < v_small_mean + v_small_std
    assert v_big_mean - v_big_std < np.mean(predicted_v_big) < v_big_mean + v_big_std

    # test that std is within +- 2x std of observations
    assert v_small_std / 2 < np.std(predicted_v_small) < v_small_std * 2
    assert v_big_std / 2 < np.std(predicted_v_big) < v_big_std * 2


if __name__ == "__main__":
    test_buoyancy_vertical_velocity(plot=True)
