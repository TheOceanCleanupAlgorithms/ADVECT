"""
advect on ECCO currents
"""
import sys
sys.path.append("../src")
from pathlib import Path

from plotting.plot_advection import animate_ocean_advection
from run_advector_3D import run_advector_3D
from datetime import datetime, timedelta

WINDAGE_MULTIPLIER = 1  # multiplier of default windage formulation (based on emerged surface area)

sourcefile = 'sourcefiles/neutral.nc'
if __name__ == '__main__':
    out_paths = run_advector_3D(
        output_directory=f'outputfiles/2015_ECCO/{Path(sourcefile).stem}/',
        sourcefile_path=sourcefile,
        configfile_path='configfiles/config.nc',
        u_water_path='ECCO/ECCO_interp/U_2015*.nc',
        v_water_path='ECCO/ECCO_interp/V_2015*.nc',
        w_water_path='ECCO/ECCO_interp/W_2015*.nc',
        u_wind_path='ncep_ncar_doe_ii/uwnd.10m.gauss.2015.nc',
        v_wind_path='ncep_ncar_doe_ii/vwnd.10m.gauss.2015.nc',
        seawater_density_path='ECCO/ECCO_interp/RHO_2015.nc',
        wind_varname_map={'uwnd': 'U', 'vwnd': 'V', 'level': 'depth'},
        advection_start_date=datetime(year=2015, month=1, day=1, hour=12),
        timestep=timedelta(hours=1),
        num_timesteps=24*365,
        save_period=24,
        advection_scheme='taylor2',
        windage_multiplier=WINDAGE_MULTIPLIER,
        wind_mixing_enabled=True,
        opencl_device=(0, 0),
        memory_utilization=.4,
    )

    for out_path in out_paths:
        animate_ocean_advection(out_path, save=False)
