from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

from tests.config import CL_CONTEXT, CL_QUEUE, MODEL_CORE_DIR

KERNEL_SOURCE = Path(__file__).parent / "test_diffusion.cl"

prg = (cl.Program(CL_CONTEXT, open(KERNEL_SOURCE).read())
       .build(options=["-I", str(MODEL_CORE_DIR)]))


def single_diffusion_step(
    z: np.ndarray,
    dt: float,
    seed: np.ndarray,
    horiz_eddy_diff_z: np.ndarray,
    horiz_eddy_diff_val: np.ndarray,
    vert_eddy_diff_z: np.ndarray,
    vert_eddy_diff_val: np.ndarray,
) -> np.ndarray:
    """do a single diffusion step at many depths (parallel), according to
        horizontal/vertical diffusivity profiles
        returns displacements, as array of 3-component vectors (xyz)"""
    out = np.zeros(len(z), dtype=cl.cltypes.double3)
    d_z = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(z))
    d_seed = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.uint32(seed))
    d_hedz = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(horiz_eddy_diff_z))
    d_hedv = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(horiz_eddy_diff_val))
    d_vedz = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(vert_eddy_diff_z))
    d_vedv = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(vert_eddy_diff_val))

    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)

    prg.single_diffusion_step(
        CL_QUEUE,
        z.shape,
        None,
        d_z,
        np.float64(dt),
        d_seed,
        d_hedz,
        d_hedv,
        np.uint32(len(horiz_eddy_diff_val)),
        d_vedz,
        d_vedv,
        np.uint32(len(vert_eddy_diff_val)),
        d_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return np.array([(v[0], v[1], v[2]) for v in out])


def test_diffusion(plot=False):
    rng = np.random.default_rng(seed=0)
    p_depth = rng.uniform(-1e4, 0, 100000)
    dt = 1  # seconds
    seed = rng.integers(1, (1 << 32) - 1, len(p_depth))  # for good random, must provide uniform seed in uint32 range

    # set up some arbitrary diffusivity profiles
    horizontal_diffusivity = np.linspace(1, 1500, 20)  # m^2 s^-1
    z_hd = -np.logspace(4, 0, 20)  # m
    vertical_diffusivity = np.linspace(5e-3, 1, 10)
    z_vd = np.linspace(-1e4, 0, 10)  # m

    drift = single_diffusion_step(p_depth, dt, seed, z_hd, horizontal_diffusivity, z_vd, vertical_diffusivity)

    z_grid = np.arange(min(p_depth), max(p_depth), 100)
    bin_radius = 100
    # centered binning for mean
    bin_mean = np.array([np.mean(drift[(p_depth > z - bin_radius) & (p_depth < z + bin_radius)], axis=0) for z in z_grid])
    # centered binning for std
    bin_std = np.array([np.std(drift[(p_depth > z - bin_radius) & (p_depth < z + bin_radius)], axis=0) for z in z_grid])
    true_std = np.array([np.sqrt(2 * np.interp(z_grid, z_hd, horizontal_diffusivity) * dt),
                         np.sqrt(2 * np.interp(z_grid, z_vd, vertical_diffusivity) * dt)])

    np.testing.assert_allclose(0, bin_mean[:, 0], atol=.2*np.mean(bin_std[:, 0]))  # mean within 1/5 O(std) of 0, p good
    np.testing.assert_allclose(0, bin_mean[:, 1], atol=.2*np.mean(bin_std[:, 1]))
    np.testing.assert_allclose(0, bin_mean[:, 2], atol=.2*np.mean(bin_std[:, 2]))
    np.testing.assert_allclose(true_std[0][1:], bin_std[:, 0][1:], rtol=.1)  # bin at bottom edge is thrown off by boundary, ignore

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(9, 9), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)
        for i, j in ((0, 0), (1, 2)):
            ax[i][0].plot(drift[:, j], p_depth, '.', markersize=.5, label='particles')
            ax[i][0].plot(np.zeros_like(z_grid), z_grid, label='expected mean', linewidth=2)
            ax[i][0].plot(bin_mean[:, j], z_grid, label='100m bin mean', linewidth=2)
            ax[i][0].plot(true_std[i], z_grid, label='expected std', linewidth=2)
            ax[i][0].plot(bin_std[:, j], z_grid, label='100m bin std', linewidth=2)
            ax[i][0].set_ylim([min(z_grid), max(z_grid)])
            ax[i][0].set_ylabel('depth (m)')

        ax[0, 0].legend()
        ax[0, 0].set_xlabel(f'lon displacement in {dt} seconds (m)')
        ax[0, 0].set_title('Depth-dependent diffusivity test')
        ax[1, 0].set_xlabel(f'depth displacement in {dt} seconds (m)')

        ax[0, 1].plot(horizontal_diffusivity, z_hd)
        ax[0, 1].set_xlabel('horizontal diffusivity ($m^2$ $s^{-1}$)')
        ax[1, 1].plot(vertical_diffusivity, z_vd)
        ax[1, 1].set_xlabel('vertical diffusivity ($m^2$ $s^{-1}$)')

if __name__ == '__main__':
    test_diffusion(plot=True)
