"""
advect on ECCO surface currents
"""

from kernel_wrappers.Kernel2D import AdvectionScheme
from plotting.plot_advection import plot_ocean_trajectories, plot_ocean_advection
from run_advector import run_advector
from datetime import datetime, timedelta

EDDY_DIFFUSIVITY = 1800  # m^2 / s
''' Sylvia Cole et al 2015: diffusivity calculated at a 300km eddy scale, global average in top 1000m, Argo float data.
  This paper shows 2 orders of magnitude variation regionally, not resolving regional differences is a big error source.
  Additionally, the assumption here is that 300km eddies are not resolved by the velocity field itself.  If they are,
  we're doubling up on the eddy transport.  For reference, to resolve 300km eddies, the grid scale probably needs to be
  on order 30km, which at the equator would be ~1/3 degree.
'''
WINDAGE_COEFF = .005  # float in [0, 1] representing fraction of wind speed that is transferred to particle
# windage coeff needs a good literature source.  Responsibility of user.  This one is taken from trashtracker.

if __name__ == '__main__':
    out_path = run_advector(
        outputfile_path='../outputfiles/2015_ECCO.nc',
        sourcefile_path='../sourcefiles/2015_uniform.nc',
        u_water_path='../forcing_data/ECCO/ECCO_interp/U*.nc',
        v_water_path='../forcing_data/ECCO/ECCO_interp/V*.nc',
        u_wind_path='../forcing_data/MERRA-2/SURFACE_WIND/*2015*.nc',
        v_wind_path='../forcing_data/MERRA-2/SURFACE_WIND/*2015*.nc',
        windfile_varname_map={'ULML': 'U', 'VLML': 'V'},
        advection_start_date=datetime(year=2015, month=1, day=1, hour=12),
        timestep=timedelta(hours=1),
        num_timesteps=24*365,
        save_period=24,
        advection_scheme=AdvectionScheme.eulerian,
        eddy_diffusivity=EDDY_DIFFUSIVITY,
        windage_coeff=WINDAGE_COEFF,
        verbose=True,
        platform_and_device=(0, 2),
    )

    plot_ocean_advection(out_path)
    #plot_ocean_trajectories(out_path)
