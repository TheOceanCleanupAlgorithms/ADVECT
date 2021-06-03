"""
advect on ECCO currents
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

examples_root = Path(__file__).parent
sys.path.append(str(examples_root.parent))
sys.path.append(str(examples_root.parent / "src"))

from examples.sourcefiles.generate_sourcefiles import generate_uniform_3D_sourcefile
from examples.configfiles.generate_configfile import generate_sample_configfile
from src.plotting.plot_advection import animate_ocean_advection
from src.run_advector_3D import run_advector_3D


if __name__ == "__main__":
    # generate a sourcefile
    sourcefile_path = examples_root / "sourcefiles/3D_uniform_source_2015.nc"
    generate_uniform_3D_sourcefile(out_path=sourcefile_path)

    # generate a configfile
    configfile_path = examples_root / "configfiles/config.nc"
    generate_sample_configfile(out_path=configfile_path)

    # run the model!
    out_paths = run_advector_3D(
        output_directory=str(
            examples_root / f"outputfiles/ECCO_2015_3D/{Path(sourcefile_path).stem}/"
        ),
        sourcefile_path=str(sourcefile_path),
        configfile_path=str(configfile_path),
        u_water_path=str(examples_root / "ECCO/currents/U_2015*.nc"),
        v_water_path=str(examples_root / "ECCO/currents/V_2015*.nc"),
        w_water_path=str(examples_root / "ECCO/currents/W_2015*.nc"),
        u_wind_path=str(examples_root / "ncep_ncar_doe_ii/uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(examples_root / "ncep_ncar_doe_ii/vwnd.10m.gauss.2015.nc"),
        wind_varname_map={"uwnd": "U", "vwnd": "V", "level": "depth"},  # wind
        seawater_density_path=str(examples_root / "ECCO/seawater_density/RHO_2015.nc"),
        advection_start_date=datetime(year=2015, month=1, day=1, hour=12),
        timestep=timedelta(hours=1),
        num_timesteps=24 * 365,
        save_period=24,
    )

    for out_path in out_paths:
        print("Animating trajectories...")
        animate_ocean_advection(out_path, save=False)
