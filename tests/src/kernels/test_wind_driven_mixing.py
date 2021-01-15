import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

from tests.config import ROOT_DIR, CL_CONTEXT, CL_QUEUE


def sample_concentration_profile(wind_10m: float, rise_velocity: float, nsamples: int) -> np.ndarray:
    """
    :return: random_depths, sampled using kukulka's concentration profile as a PDF
    """
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "wind_driven_mixing.cl"
    #include "random.cl"

    __kernel void test_sample_concentration_profile(
        double wind_10m,
        double rise_velocity,
        unsigned int nsamples,
        __global double *out) {
        random_state rstate = {.a = 1};
        for (unsigned int i=0; i<nsamples; i++) {
            random(&rstate);
            out[i] = sample_concentration_profile(wind_10m, rise_velocity, &rstate);
        }
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(nsamples).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_sample_concentration_profile(
            CL_QUEUE, (1,), None,
            np.float64(wind_10m),
            np.float64(rise_velocity),
            np.uint32(nsamples),
            d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_sample_concentration_profile(plot=False):
    """tries to reproduce concentration plot given in kukulka 2012 fig 3a
        parameters: 10m wind speed is 6.5 m/s (based on the .75 cm/s water friction, stated average conditions)
                    particle rise velocity is modeled at .01 m/s, as they do.
    """
    u10 = 6.5  # m/s, roughly corresponds to water friction velocity .75 cm/s
    w_b = .01  # m s^-1, given rise velocity
    A_0 = .004443  # computed from ustar=.75, or u10 = 6.5 m/s
    samples = sample_concentration_profile(wind_10m=u10, rise_velocity=w_b, nsamples=100000)
    measured_PDF, bin_edges = np.histogram(samples, bins=50, range=(-3, 0), density=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    true_PDF = w_b / A_0 * np.exp(w_b / A_0 * bin_centers)
    np.testing.assert_allclose(true_PDF, measured_PDF, atol=.02)  # every bin within 2%, that's a super close match
    if plot:
        plt.figure()
        plt.plot(true_PDF, bin_centers, label='PDF')
        plt.plot(measured_PDF, bin_centers, label='measured PDF')
        plt.hist(samples, bins=100, label='samples', orientation='horizontal', density=True)
        plt.ylim([-3, 0])
        plt.legend()
        plt.ylabel('z')


def calculate_wind_stress(wind_speed_10m: float) -> float:
    """
    :param wind_speed_10m: m/s
    :return: wind stress (kg m^-1 s^-2)
    """
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "wind_driven_mixing.cl"

    __kernel void test_calculate_wind_stress(
        const double wind_speed_10m,
        __global double *out) {

        out[0] = calculate_wind_stress(wind_speed_10m);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_calculate_wind_stress(
            CL_QUEUE, (1,), None,
            np.float64(wind_speed_10m),
            d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out[0]


def test_calculate_wind_stress():
    """check matches some examples from kukulka"""
    # kukulka says for u10 = 4.7 m/s, water_friction_velocity = .55 cm/s.
    # water_friction_velocity = sqrt(wind_stress/water_density), so we can test this to compare methods.
    u10 = 4.7
    wind_stress = calculate_wind_stress(u10)
    water_density = 1025  # kg m^-3

    kukulka_result = .55e-2  # m/s
    np.testing.assert_allclose(kukulka_result, np.sqrt(wind_stress/water_density), rtol=.01)


if __name__ == '__main__':
    test_sample_concentration_profile(plot=True)
