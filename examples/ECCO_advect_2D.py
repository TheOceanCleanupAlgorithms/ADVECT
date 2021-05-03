"""
advect on HYCOM surface currents
"""
import glob
import sys
from pathlib import Path

examples_root = Path(__file__).parent
sys.path.append(str(examples_root.parent / "src"))
from run_advector_2D import run_advector_2D
from plotting.plot_advection import animate_ocean_advection, plot_ocean_trajectories
from datetime import datetime, timedelta

ECCO_U_PATH = str(examples_root / 'ECCO/ECCO_interp/U_2015*.nc')
ECCO_V_PATH = str(examples_root / 'ECCO/ECCO_interp/V_2015*.nc')

if __name__ == '__main__':
    sourcefile = str(examples_root / "sourcefiles/2D_uniform_source_2015.nc")
    ADVECTION_START = datetime(2015, 1, 1)
    ADVECTION_END = datetime(2016, 1, 1)
    out_paths = run_advector_2D(
        output_directory=str(examples_root / "outputfiles/ECCO_2015_2D" / Path(sourcefile).stem),
        sourcefile_path=sourcefile,
        u_water_path=ECCO_U_PATH,
        v_water_path=ECCO_V_PATH,
        u_wind_path=str(examples_root / 'ncep_ncar_doe_ii/uwnd.10m.gauss.2015.nc'),
        v_wind_path=str(examples_root / 'ncep_ncar_doe_ii/vwnd.10m.gauss.2015.nc'),
        wind_varname_map={'uwnd': 'U', 'vwnd': 'V', 'level': 'depth'},
        windage_coeff=.005,
        eddy_diffusivity=200,  # m^2 / s
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24*(ADVECTION_END - ADVECTION_START).days,
        save_period=24,
        memory_utilization=.4,  # decrease if RAM overloaded.  Can be close to 1 on dedicated compute device (e.g. GPU)
        opencl_device=(0, 0),
    )

    for path in out_paths:
        animate_ocean_advection(outputfile_path=path, save=False)
        plot_ocean_trajectories(path, glob.glob(ECCO_U_PATH)[0])
