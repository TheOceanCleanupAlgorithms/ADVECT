"""
advect on ECCO currents
"""
from pathlib import Path

from plotting.plot_advection import animate_ocean_advection
from run_advector import run_advector
from datetime import datetime, timedelta

EDDY_DIFFUSIVITY = 0  # m^2 / s
# needs a good literature source.  Responsibility of user.
WINDAGE_MULTIPLIER = 1  # multiplier of default windage formulation (based on emerged surface area)

sourcefile = 'sourcefiles/neutral.nc'
if __name__ == '__main__':
    out_paths = run_advector(
        output_directory=f'outputfiles/2015_ECCO/{Path(sourcefile).stem}/',
        sourcefile_path=sourcefile,
        u_water_path='ECCO/ECCO_interp/U_2015*.nc',
        v_water_path='ECCO/ECCO_interp/V_2015*.nc',
        w_water_path='ECCO/ECCO_interp/W_2015*.nc',
        u_wind_path='MERRA2_wind/*2015*.nc',
        v_wind_path='MERRA2_wind/*2015*.nc',
        wind_varname_map={'ULML': 'U', 'VLML': 'V'},
        advection_start_date=datetime(year=2015, month=1, day=1, hour=12),
        timestep=timedelta(hours=1),
        num_timesteps=24*365,
        save_period=24,
        advection_scheme='taylor2',
        eddy_diffusivity=EDDY_DIFFUSIVITY,
        windage_multiplier=WINDAGE_MULTIPLIER,
        verbose=True,
        opencl_device=(0, 2),
        memory_utilization=.95,
    )

    for out_path in out_paths:
        animate_ocean_advection(out_path, save=False)
