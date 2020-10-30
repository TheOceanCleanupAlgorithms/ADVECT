"""
advect on ECCO surface currents
"""

from kernel_wrappers.Kernel2D import AdvectionScheme
from plotting.plot_advection import plot_ocean_advection, plot_ocean_trajectories
from run_advector import run_advector

#EDDY_DIFFUSIVITY = 1800  # m^2 / s
EDDY_DIFFUSIVITY = 0  # m^2 / s
''' Sylvia Cole et al 2015: diffusivity calculated at a 300km eddy scale, global average in top 1000m, Argo float data.
  This paper shows 2 orders of magnitude variation regionally, not resolving regional differences is a big error source.
  Additionally, the assumption here is that 300km eddies are not resolved by the velocity field itself.  If they are,
  we're doubling up on the eddy transport.  For reference, to resolve 300km eddies, the grid scale probably needs to be
  on order 30km, which at the equator would be ~1/3 degree.
'''


if __name__ == '__main__':
    out_path = run_advector(
        outputfile_path='../outputfiles/2015_ECCO.nc',
        sourcefile_path='../sourcefiles/small.nc',
        u_path='../forcing_data/ECCO/ECCO_interp/U*.nc',
        v_path='../forcing_data/ECCO/ECCO_interp/V*.nc',
        advection_start='2015-01-01T12',
        timestep_seconds=3600,
        num_timesteps=24*365,
        save_period=24,
        advection_scheme=AdvectionScheme.eulerian,
        eddy_diffusivity=EDDY_DIFFUSIVITY,
        verbose=True,
    )

    plot_ocean_trajectories(out_path)
