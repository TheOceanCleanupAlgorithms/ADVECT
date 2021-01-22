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
    rho = rng.uniform(930, 970, nsamples)  # plastic density not stated; this mimics a selection of LDPE and HDPE

    # in figure 2, we see pictures of fragments.  they all look like wafers, but smaller ones are more rounded.
    corey_shape_factor_small = .5/np.sqrt(1*1)  # figure 2a, looks like fragments are twice as long as they are thick
    corey_shape_factor_big = 1/np.sqrt(4*4)# figure 2b, looks like fragments are ~4x as long as they are thick

    # "size" is given.  Given that they used seives with mesh size of their "size classes", and the particles apear to
    # be wafers, it is reasonable to say that "size" refers to the LONG dimension.
    # as such, we can estimate radius by pretending the volume of the wafer was shaped as a sphere.
    size_small = np.linspace(.5e-3, 1.5e-3, nsamples)
    volume_small = (.5*size_small)*size_small**2  # we assume short dimension is .5 of long dimension
    r_small = np.cbrt(3/(4*np.pi) * volume_small)
    v_small_mean = .009  # m/s
    v_small_std = .004   # m/s

    size_big = np.linspace(1.5e-3, 5e-3, nsamples)  # again, size means the long dimension
    volume_big = (.25*size_big)*size_big**2  # now we assume short dimension is .25 of long dimension
    r_big = np.cbrt(3/(4*np.pi) * volume_big)
    v_big_mean = .019  # m/s
    v_big_std = .006   # m/s

    predicted_v_small = buoyancy_vertical_velocity(rho, r_small, np.full_like(rho, corey_shape_factor_small))
    predicted_v_big = buoyancy_vertical_velocity(rho, r_big, np.full_like(rho, corey_shape_factor_big))

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(np.concatenate([size_small, size_big])*1e3, np.concatenate([predicted_v_small, predicted_v_big]), '.', label='predictions (rho ~ U(930, 970))', color='tab:cyan', markersize=2)
        plt.errorbar(np.mean(size_small)*1e3-.05, np.mean(predicted_v_small), np.std(predicted_v_small), capsize=3, color='tab:blue', zorder=3)
        plt.errorbar(np.mean(size_big)*1e3-.05, np.mean(predicted_v_big), np.std(predicted_v_big), capsize=3, color='tab:blue', label='prediction distribution', zorder=3)
        plt.errorbar(np.mean(size_small)*1e3+.05, v_small_mean, v_small_std, capsize=3, color='tab:green', zorder=3)
        plt.errorbar(np.mean(size_big)*1e3+.05, v_big_mean, v_big_std, capsize=3, color='tab:green', label='observed distribution (Kooi 2016)', zorder=3)
        plt.xlabel("Fragment size (mm)")
        plt.ylabel("Rise velocity (m/s)")
        plt.legend()
        plt.title("Theoretical rise velocity vs observations")

    # test that mean is within 1.5 std of obvervations
    assert v_small_mean - 1.5*v_small_std < np.mean(predicted_v_small) < v_small_mean + v_small_std*1.5
    assert v_big_mean - 1.5*v_big_std < np.mean(predicted_v_big) < v_big_mean + v_big_std*1.5

    # test that std is within multiple of 2 of std of observations
    assert v_small_std / 2 < np.std(predicted_v_small) < v_small_std * 2
    assert v_big_std / 2 < np.std(predicted_v_big) < v_big_std * 2


if __name__ == "__main__":
    test_buoyancy_vertical_velocity(plot=True)
