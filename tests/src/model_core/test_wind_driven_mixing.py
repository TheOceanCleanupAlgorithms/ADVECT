import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

from tests.config import CL_CONTEXT, CL_QUEUE, MODEL_CORE_DIR


def sample_concentration_profile(
    wind_10m: float, rise_velocity: float, nsamples: int
) -> np.ndarray:
    """
    :return: random_depths, sampled using kukulka's concentration profile as a PDF
    """
    # setup
    prg = cl.Program(
        CL_CONTEXT,
        """
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
            out[i] = sample_concentration_profile(wind_10m, rise_velocity, 20, 10, &rstate);
        }
    }
    """,
    ).build(options=["-I", str(MODEL_CORE_DIR)])

    out = np.zeros(nsamples).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_sample_concentration_profile(
        CL_QUEUE,
        (1,),
        None,
        np.float64(wind_10m),
        np.float64(rise_velocity),
        np.uint32(nsamples),
        d_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_sample_concentration_profile(plot=False):
    """tries to reproduce concentration plot given in Kukulka 2012 fig 3a
    parameters: 10m wind speed is 6.5 m/s (based on the .75 cm/s water friction, stated average conditions)
                particle rise velocity is modeled at .01 m/s, as in Kukulka
    """
    u10 = 6.5  # m/s, roughly corresponds to water friction velocity .75 cm/s
    w_b = 0.01  # m s^-1, given rise velocity
    A_0 = 0.002628  # computed from ustar=.75, or u10 = 6.5 m/s
    H_s = calculate_significant_wave_height(
        calculate_wind_stress(u10)
    )  # significant wave height
    MLD = -10 * H_s  # as defined in wind_driven_mixing.cl

    samples = sample_concentration_profile(
        wind_10m=u10, rise_velocity=w_b, nsamples=100000
    )
    measured_PDF, bin_edges = np.histogram(
        samples, bins=50, range=(MLD, 0), density=True
    )
    z = bin_edges[:-1] + np.diff(bin_edges) / 2
    true_PDF = w_b / (A_0 * (1 - np.exp(w_b / A_0 * MLD))) * np.exp(w_b / A_0 * z)

    def test_and_plot():
        if plot:
            plt.figure()
            plt.plot(true_PDF, z, label="PDF")
            plt.plot(measured_PDF, z, label="measured PDF")
            plt.hist(
                samples,
                bins=50,
                label="samples",
                range=(MLD, 0),
                orientation="horizontal",
                density=True,
            )
            plt.ylim([MLD, 0])
            plt.legend()
            plt.ylabel("z")
            plt.title(f"w_b={w_b} m/s, u10={u10} m/s --> MLD={MLD: .2f} m")
        np.testing.assert_allclose(
            true_PDF, measured_PDF, atol=0.03
        )  # every bin within 3%, that's a super close match

    test_and_plot()

    # now try with a neutral particle (special case, as PDF/CDF become undefined
    #   but asymptotic behavior is understood to be a uniform distribution)
    w_b = 0

    samples = sample_concentration_profile(
        wind_10m=u10, rise_velocity=w_b, nsamples=100000
    )
    measured_PDF, bin_edges = np.histogram(
        samples, bins=50, range=(MLD, 0), density=True
    )
    z = bin_edges[:-1] + np.diff(bin_edges) / 2
    true_PDF = np.ones_like(z) / (0 - MLD)
    test_and_plot()


def calculate_significant_wave_height(wind_stress: float) -> float:
    """
    :param wind_stress (kg m^-1 s^-2)
    :return: significant wave height (m)
    """
    # setup
    prg = cl.Program(
        CL_CONTEXT,
        """
    #include "wind_driven_mixing.cl"

    __kernel void test_calculate_significant_wave_height(
        const double wind_stress,
        __global double *out) {

        out[0] = calculate_significant_wave_height(wind_stress, 20);
    }
    """,
    ).build(options=["-I", str(MODEL_CORE_DIR)])

    out = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_calculate_significant_wave_height(
        CL_QUEUE, (1,), None, np.float64(wind_stress), d_out
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out[0]


def test_calculate_significant_wave_height():
    # kukulka says for u10 = ~6 m/s, water_friction_velocity = ~.75 cm/s.
    # we can get wind stress from this, and calculate wave height, to at least do a sanity check
    water_density = 1025  # kg m^-3
    wind_stress = (
        0.75e-2 ** 2 * water_density
    )  # m/s  inverting kukulka's given frictional water velocity
    wave_height = calculate_significant_wave_height(wind_stress)
    assert 0 < wave_height < 1  # sanity range check

    # check the maximum allowable wave height is not surpassed in absurd winds
    # max height defined in physical_constants.h
    assert (
        calculate_significant_wave_height(calculate_wind_stress(wind_speed_10m=40))
        == 20
    )


def calculate_wind_stress(wind_speed_10m: float) -> float:
    """
    :param wind_speed_10m: m/s
    :return: wind stress (kg m^-1 s^-2)
    """
    # setup
    prg = cl.Program(
        CL_CONTEXT,
        """
    #include "wind_driven_mixing.cl"

    __kernel void test_calculate_wind_stress(
        const double wind_speed_10m,
        __global double *out) {

        out[0] = calculate_wind_stress(wind_speed_10m);
    }
    """,
    ).build(options=["-I", str(MODEL_CORE_DIR)])

    out = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_calculate_wind_stress(
        CL_QUEUE, (1,), None, np.float64(wind_speed_10m), d_out
    )
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

    kukulka_result = 0.55e-2  # m/s
    np.testing.assert_allclose(
        kukulka_result, np.sqrt(wind_stress / water_density), rtol=0.01
    )


if __name__ == "__main__":
    test_sample_concentration_profile(plot=True)
