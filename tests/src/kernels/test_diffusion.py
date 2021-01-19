import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tests.config import ROOT_DIR, CL_CONTEXT, CL_QUEUE

KERNEL_SOURCE = Path(__file__).parent / "test_diffusion.cl"

prg = (cl.Program(CL_CONTEXT, open(KERNEL_SOURCE).read())
       .build(options=["-I", str(ROOT_DIR / "src/kernels")]))


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
    d_seed = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(seed))
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

    result = single_diffusion_step(p_depth, dt, seed, z_hd, horizontal_diffusivity, z_vd, vertical_diffusivity)

    drift = np.abs(result)

    z_grid = np.arange(min(p_depth), max(p_depth), 100)
    bin_radius = 50
    # centered binning for mean
    bin_mean = np.array([np.mean(drift[(p_depth > z - bin_radius) & (p_depth < z + bin_radius)], axis=0) for z in z_grid])
    # non-centered binning for max
    bin_max = np.array([np.max(drift[(p_depth > z - 2 * bin_radius) & (p_depth < z)], axis=0) for z in z_grid[1:]])
    diff_amp_horiz = np.sqrt(4 * np.interp(z_grid, z_hd, horizontal_diffusivity) * dt)
    expected_step_horiz = diff_amp_horiz * .5  # expected value of a (0,1) uniform distribution is .5
    diff_amp_vert = np.sqrt(2 * np.interp(z_grid, z_vd, vertical_diffusivity) * dt)
    expected_step_vert = diff_amp_vert * .5  # expected value of a (0,1) uniform distribution is .5

    # within a bin radius from the edge, the bin mean skews because no data past domain
    np.testing.assert_allclose(bin_mean[:, 0][z_grid > min(z_grid) + bin_radius],
                               expected_step_horiz[z_grid > min(z_grid) + bin_radius], rtol=.1)
    np.testing.assert_allclose(bin_mean[:, 1][z_grid > min(z_grid) + bin_radius],
                               expected_step_horiz[z_grid > min(z_grid) + bin_radius], rtol=.1)
    np.testing.assert_allclose(bin_mean[:, 2][z_grid > min(z_grid) + bin_radius],
                               expected_step_vert[z_grid > min(z_grid) + bin_radius], rtol=.1)
    np.testing.assert_array_less(bin_max[:, 0], diff_amp_horiz[1:])
    np.testing.assert_array_less(bin_max[:, 1], diff_amp_horiz[1:])
    np.testing.assert_array_less(bin_max[:, 2], diff_amp_vert[1:])

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        ax[0].plot(drift[:, 0], p_depth, '.', markersize=.5, label='particles')
        ax[0].plot(bin_mean[:, 0], z_grid, label='100m bin mean')
        ax[0].plot(bin_max[:, 0], z_grid[1:], label='100m bin max')
        ax[0].plot(expected_step_horiz, z_grid, label='expected diffusivity step (m)')
        ax[0].plot(diff_amp_horiz, z_grid, label='max diffusivity amplitude (m)')
        ax[0].set_ylim([min(z_grid), max(z_grid)])
        ax[0].set_ylabel('depth (m)')
        ax[0].set_xlabel(f'lon displacement in {dt} seconds (m)')
        ax[0].legend()
        ax[0].set_title('Depth-dependent diffusivity test')
        ax[1].plot(drift[:, 2], p_depth, '.', markersize=.5, label='particles')
        ax[1].plot(bin_mean[:, 2], z_grid, label='100m bin mean')
        ax[1].plot(bin_max[:, 2], z_grid[1:], label='100m bin max')
        ax[1].plot(expected_step_vert, z_grid, label='expected diffusivity step (m)')
        ax[1].plot(diff_amp_vert, z_grid, label='max diffusivity amplitude (m)')
        ax[1].set_ylim([min(z_grid), max(z_grid)])
        ax[1].set_ylabel('depth (m)')
        ax[1].set_xlabel(f'depth displacement in {dt} seconds (m)')


if __name__ == '__main__':
    test_diffusion(plot=True)
