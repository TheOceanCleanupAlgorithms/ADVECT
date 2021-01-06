import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyopencl as cl

from io_tools.open_vectorfiles import empty_2D_vectorfield
from kernel_wrappers.Kernel3D import Kernel3D, AdvectionScheme
from tests.src.kernels.test_geography import degrees_lat_to_meters


def run_diffusion(num_particles: int):
    """run a bunch of particles through one timestep; no current, no winds, and a set horizontal diffusivity profile"""
    # some arbitrary profile
    horizontal_eddy_diffusivity = xr.Dataset(
            {"horizontal_diffusivity": ("z_hd", (np.linspace(1, 1500, 20)))},
            coords={"z_hd": -np.logspace(4, 0, 20)}
    )

    current_depth = np.linspace(-10000, 0, 100)
    current = xr.Dataset(
        {
            "U": (["lat", "lon", "depth", "time"], np.zeros((1, 1, len(current_depth), 1))),
            "V": (["lat", "lon", "depth", "time"], np.zeros((1, 1, len(current_depth), 1))),
            "W": (["lat", "lon", "depth", "time"], np.zeros((1, 1, len(current_depth), 1))),
        },
        coords={
            "lon": [0],
            "lat": [0],
            "depth": current_depth,
            "time": [np.datetime64('2000-01-01')],
        },
    )
    rng = np.random.default_rng(seed=0)
    p0 = pd.DataFrame({'lon': np.zeros(num_particles), 'lat': np.zeros(num_particles), 'p_id': np.arange(num_particles),
                       'depth': rng.uniform(min(current_depth), max(current_depth), num_particles),
                       'radius': .001*np.ones(num_particles), 'density': 1025 * np.ones(num_particles), 'exit_code': np.zeros(num_particles)})

    time = pd.date_range(start='2000-01-01', periods=2, freq='1s')
    p0['release_date'] = time[0]
    p0 = xr.Dataset(p0.set_index('p_id'))

    P = Kernel3D(current=current, wind=empty_2D_vectorfield(), p0=p0,
                 advect_time=time, save_every=1,
                 advection_scheme=AdvectionScheme.eulerian, eddy_diffusivity=horizontal_eddy_diffusivity,
                 windage_multiplier=None, context=cl.create_some_context()).execute().squeeze()
    return P, horizontal_eddy_diffusivity


def test_diffusion(plot=False):
    P, config = run_diffusion(num_particles=100000)

    timestep_seconds = 1
    drift = degrees_lat_to_meters(deg_lat=np.abs(P.lat.values), lat=0)
    p_depth = P.depth.values

    z_grid = np.arange(P.depth.min(), P.depth.max(), 100)
    bin_radius = 50
    # centered binning for mean
    bin_mean = np.array([np.mean(drift[(p_depth > z - bin_radius) & (p_depth < z + bin_radius)]) for z in z_grid])
    # non-centered binning for max
    bin_max = np.array([np.max(drift[(p_depth > z - 2*bin_radius) & (p_depth < z)]) for z in z_grid[1:]])
    diff_amp = np.sqrt(4 * config.horizontal_diffusivity.interp(z_hd=z_grid, method='linear') * timestep_seconds)
    expected_step = diff_amp * .5  # expected value of a (0,1) uniform distribution is .5

    # within a bin radius from the edge, the bin mean skews because no data past domain
    np.testing.assert_allclose(bin_mean[z_grid > min(z_grid)+bin_radius], expected_step[z_grid > min(z_grid)+bin_radius], rtol=.1)
    np.testing.assert_array_less(bin_max, diff_amp[1:])

    if plot:
        fig, ax = plt.subplots(1)
        ax.plot(drift, p_depth, '.', markersize=.5, label='particles')
        ax.plot(bin_mean, z_grid, label='100m bin median')
        ax.plot(bin_max, z_grid[1:], label='100m bin max')
        ax.plot(expected_step, z_grid, label='expected diffusivity step (m)')
        ax.plot(diff_amp, z_grid, label='max diffusivity amplitude (m)')
        ax.set_ylim([min(z_grid), max(z_grid)])
        ax.set_ylabel('depth (m)')
        ax.set_xlabel(f'lat displacement in {timestep_seconds} seconds (m)')
        ax.legend()
        ax.set_title('Depth-dependent horizontal diffusivity test')


if __name__ == '__main__':
    test_diffusion(plot=True)
