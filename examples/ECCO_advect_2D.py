"""
advect on ECCO surface currents
"""
import glob
import sys
import pandas as pd
from pathlib import Path


examples_root = Path(__file__).parent
sys.path.append(str(examples_root))
sys.path.append(str(examples_root.parent / "src"))


from sourcefiles.generate_sourcefiles import generate_2D_sourcefile
from run_advector_2D import run_advector_2D
from plotting.plot_advection import animate_ocean_advection, plot_ocean_trajectories
from datetime import datetime, timedelta


ECCO_U_PATH = examples_root / "ECCO/currents/U_2015*.nc"
ECCO_V_PATH = examples_root / "ECCO/currents/V_2015*.nc"

if __name__ == "__main__":
    sourcefile_path = examples_root / "sourcefiles/2D_uniform_source_2015.nc"
    generate_2D_sourcefile(
        num_particles=5000,
        release_date_range=(pd.Timestamp(2015, 1, 1), pd.Timestamp(2015, 12, 31)),
        out_path=sourcefile_path,
    )
    ADVECTION_START = datetime(2015, 1, 1)
    ADVECTION_END = datetime(2016, 1, 1)
    out_paths = run_advector_2D(
        output_directory=str(
            examples_root / f"outputfiles/ECCO_2015_2D/{sourcefile_path.stem}"
        ),
        sourcefile_path=str(sourcefile_path),
        u_water_path=str(ECCO_U_PATH),
        v_water_path=str(ECCO_V_PATH),
        u_wind_path=str(examples_root / "ncep_ncar_doe_ii/uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(examples_root / "ncep_ncar_doe_ii/vwnd.10m.gauss.2015.nc"),
        wind_varname_map={"uwnd": "U", "vwnd": "V", "level": "depth"},  # wind
        windage_coeff=0.005,
        eddy_diffusivity=200,  # m^2 / s
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24 * (ADVECTION_END - ADVECTION_START).days,
        save_period=24,
        memory_utilization=0.4,  # decrease if RAM overloaded.  Can be close to 1 on dedicated compute device (e.g. GPU)
    )

    for path in out_paths:
        animate_ocean_advection(outputfile_path=path, save=False)
        plot_ocean_trajectories(path, glob.glob(str(ECCO_U_PATH))[0])
