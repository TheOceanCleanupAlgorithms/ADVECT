import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

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


def test_kooi_2016(plot=False):
    """check theory against observations in Kooi 2016, figure 4"""
    nsamples = 1000
    rng = np.random.default_rng(seed=1)
    rho_min, rho_max = 930, 970
    rho = rng.uniform(rho_min, rho_max, nsamples)  # plastic density not stated; this mimics a selection of LDPE and HDPE

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
        plt.plot(np.concatenate([size_small, size_big])*1e3, np.concatenate([predicted_v_small, predicted_v_big]),
                 '.', label=f'predictions (rho ~ U({rho_min}, {rho_max}) kg m^-3)', color='tab:cyan', markersize=2)
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


def compare_lebreton_2018(plot=False):
    """compare theory against observations in Lebreton 2018, figure S2, type H"""
    # plot data from lebreton 2018
    size_classes = np.array([
            [.05, .15],
            [.15, .5],
            [.5, 1.5],
            [1.5, 5],
            [5, 10],
            [10, 50],
            [50, 100],  # technically upper bound undefined
    ]) * 1e-2  # units: cm (raw) converted to m
    # boxplots extracted from figure S2, type H
    boxplot_shapes = np.array([  # [lower whisker, 1st quartile, median, 3rd quartile, upper whisker]
            [0.220, 0.907, 1.310, 1.570, 2.446],
            [0.925, 1.687, 2.208, 2.587, 3.630],
            [1.617, 2.730, 3.606, 5.170, 7.526],
            [1.430, 2.732, 4.469, 7.109, 13.580],
            [1.317, 3.236, 5.248, 6.884, 12.009],
            [2.845, 5.312, 6.809, 9.585, 14.922],
            [3.479, 6.841, 9.096, 12.299, 16.290]
    ]) * 1e-2  # units: cm s^-1 (raw) converted to m s^-1

    # now we'll generate a sample of plastic that seems similar enough.
    p = {}
    nsamples = 100000
    rng = np.random.default_rng(seed=1)
    rho_min, rho_max = 930, 970
    p['rho'] = rng.uniform(rho_min, rho_max, nsamples)  # plastic density not stated; this mimics a selection of LDPE and HDPE

    # we're going to generate solid rectangular prisms; we'll generate all kinds of shapes and sizes.
    p['dims'] = np.sort(
        rng.lognormal(mean=np.mean(np.log(size_classes/2)), sigma=4, size=(nsamples, 3)),
        axis=1,
    )

    # from the dimensions, we can calculate a shape factor and a nominal radius.
    p['csf'] = p['dims'][:, 0] / np.sqrt(p['dims'][:, 1] * p['dims'][:, 2])
    p['r'] = np.cbrt(
        3 / (4 * np.pi) * np.prod(p['dims'], axis=1)
    )  # calculate p volumes, then get radius of equivalent sphere

    # now we need to reject particles with CSF <= .15 (because that's outside our model domain),
    # and dims outside the boxplot domain (from the lognormal size picking)
    valid_mask = (
        (p['csf'] > .15) &
        (np.max(p['dims'], axis=1) < np.max(size_classes)) &
        (np.min(p['dims'], axis=1) > np.min(size_classes))
    )
    for key, value in p.items():
        p[key] = p[key][valid_mask]

    # and we can calculate rise velocity for all our particles!
    p['rise_velocity'] = buoyancy_vertical_velocity(p['rho'], p['r'], p['csf'])

    # now, we need to calculate a "size class" of our particle.  I think longest dimension is probably best for this.
    p['size_class'] = p['dims'][:, 2]

    # now we separate into size classes:
    rise_velocity_binned = []
    for bnds in size_classes:
        rise_velocity_binned.append(p['rise_velocity'][(bnds[0] < p['size_class']) & (p['size_class'] < bnds[1])])

    if plot:
        fig, ax = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(8, 6))
        ax[0].bxp(
            bxpstats=[
                {"whislo": whislo, "q1": q1, "med": med, "q3": q3, "whishi": whishi}
                for (whislo, q1, med, q3, whishi) in boxplot_shapes * 1e2
            ],
            positions=size_classes.mean(axis=1) * 1e2,
            widths=np.diff(size_classes, axis=1) * 1e2,
            showfliers=False,
        )
        ax[0].set_title('Boxplot from Lebreton 2018 figure S2, plastic type H')

        ax[1].boxplot(
            x=[rv * 1e2 for rv in rise_velocity_binned],
            positions=size_classes.mean(axis=1) * 1e2,
            widths=np.diff(size_classes, axis=1) * 1e2,
            sym="",
        )
        ax[1].set_title('Boxplot of Modeled Particles')
        for axis in ax:
            axis.set_ylabel('rise velocity (cm $s^{-1}$)')
            axis.plot(p['size_class'] * 1e2, p['rise_velocity'] * 1e2, '.', alpha=.1, markersize=3, label='modeled particles')
        ax[0].plot(size_classes.mean(axis=1) * 1e2, [1e2 * np.median(rv) for rv in rise_velocity_binned], '--', color='tab:blue', label='modeled median')
        ax[0].legend()
        ax[1].set_xlabel('size class (cm)')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')


if __name__ == "__main__":
    test_kooi_2016(plot=True)
    compare_lebreton_2018(plot=True)
