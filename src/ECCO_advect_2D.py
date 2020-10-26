"""
advect on ECCO surface currents
"""

import xarray as xr
from kernel_wrappers.Kernel2D import AdvectionScheme
from plotting.plot_advection import plot_ocean_advection
from run_advector import run_advector

if __name__ == '__main__':
    out_path = run_advector(
        output_dir='../outputfiles/',
        sourcefile_path='../sourcefiles/2015_uniform.nc',
        u_path='../forcing_data/ECCO/ECCO_interp/U*.nc',
        v_path='../forcing_data/ECCO/ECCO_interp/V*.nc',
        advection_start='2015-01-01T12',
        timestep_seconds=3600,
        num_timesteps=24*365,
        save_period=24,
        advection_scheme=AdvectionScheme.taylor2,
        verbose=True
    )
    P = xr.open_dataset(out_path)
    plot_ocean_advection(P)
