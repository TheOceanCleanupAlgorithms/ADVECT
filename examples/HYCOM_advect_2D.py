"""
advect on HYCOM surface currents
"""
from run_advector import run_advector
from plotting.plot_advection import plot_ocean_advection, plot_ocean_trajectories
from datetime import datetime, timedelta


U_WATER_PATH = "./HYCOM_currents/uv*2015*.nc"
V_WATER_PATH = "./HYCOM_currents/uv*2015*.nc"
U_WIND_PATH = "./MERRA2_wind/*2015*.nc"
V_WIND_PATH = "./MERRA2_wind/*2015*.nc"
SOURCEFILE_PATH = "./sourcefiles/2015_uniform_two_releases.nc"
OUTPUT_FOLDER = "./outputfiles/HYCOM_2015/"

ADVECTION_START = datetime(2015, 1, 1)
ADVECTION_END = datetime(2016, 1, 1)

EDDY_DIFFUSIVITY = 0  # m^2 / s, user determined
WINDAGE_COEFF = .005  # fraction of wind speed transferred to particle, user determined


if __name__ == '__main__':
    out_paths = run_advector(
        output_directory=OUTPUT_FOLDER,
        sourcefile_path=SOURCEFILE_PATH,
        u_water_path=U_WATER_PATH,
        v_water_path=V_WATER_PATH,
        water_varname_map={'water_u': 'U', 'water_v': 'V'},
        # u_wind_path=U_WIND_PATH,      # uncomment these if you wish to enable windage
        # v_wind_path=V_WIND_PATH,
        # wind_varname_map={'ULML': 'U', 'VLML': 'V'},
        # windage_coeff=WINDAGE_COEFF,
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24*(ADVECTION_END - ADVECTION_START).days,
        save_period=24,
        eddy_diffusivity=EDDY_DIFFUSIVITY,
        verbose=True,
        # sourcefile_format="trashtracker",  # uncomment for trashtracker source files
        memory_utilization=.4,  # decrease if RAM overloaded.  Can be close to 1 on dedicated compute device (e.g. GPU)
    )

    for path in out_paths:
        plot_ocean_advection(path)
        plot_ocean_trajectories(path)
