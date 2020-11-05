"""
advect on HYCOM surface currents
"""
import os

from kernel_wrappers.Kernel2D import AdvectionScheme
from run_advector import run_advector
from plotting.plot_advection import plot_ocean_advection
from io_tools.open_sourcefiles import SourceFileType
from datetime import datetime, timedelta


U_WATER_PATH = os.path.join(os.path.dirname(__file__), "currents/uv*.nc")
V_WATER_PATH = os.path.join(os.path.dirname(__file__), "currents/uv*.nc")
SOURCEFILE_PATH = os.path.join(os.path.dirname(__file__), "../sourcefiles/2015_uniform_two_releases.nc")
OUTPUTFILE_PATH = os.path.join(os.path.dirname(__file__), "../outputfiles/HYCOM_2015_out.nc")

ADVECTION_START = datetime(2015, 1, 1)
ADVECTION_END = datetime(2016, 1, 1)

EDDY_DIFFUSIVITY = 0  # m^2 / s, user determined


if __name__ == '__main__':
    out_path = run_advector(
        outputfile_path=OUTPUTFILE_PATH,
        sourcefile_path=SOURCEFILE_PATH,
        u_water_path=U_WATER_PATH,
        v_water_path=V_WATER_PATH,
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24*(ADVECTION_END - ADVECTION_START).days,
        save_period=24,
        advection_scheme=AdvectionScheme.taylor2,
        eddy_diffusivity=EDDY_DIFFUSIVITY,
        platform_and_device=None,  # requests user input
        currents_varname_map={'water_u': 'U', 'water_v': 'V'},
        verbose=True,
        source_file_type=SourceFileType.new_source_files,  # .old_source_files for trashtracker source files
        # sourcefile_varname_map={'release_date': 'release_date', 'x': 'id'},  # for trashtracker source files
        memory_utilization=.5,  # decrease if RAM overloaded.  Can be close to 1 on dedicated compute device (e.g. GPU)
    )

    plot_ocean_advection(out_path)
