"""
advect on ECCO currents
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

examples_root = Path(__file__).parent
sys.path.append(str(examples_root.parent))
sys.path.append(str(examples_root.parent / "src"))

from helpers.generate_sourcefiles import generate_uniform_3D_sourcefile
from helpers.generate_configfile import generate_sample_configfile
from src.plotting.plot_advection import animate_ocean_advection
from src.run_advector_3D import run_advector_3D


if __name__ == "__main__":
    data_root = Path(input("Input path to example data directory: "))
    output_root = Path(input("Input path to directory for outputfiles: "))
    output_root.mkdir(exist_ok=True)
    # generate a sourcefile
    sourcefile_path = output_root / "3D_uniform_source_2015.nc"
    generate_uniform_3D_sourcefile(out_path=sourcefile_path)

    # generate a configfile
    configfile_path = output_root / "config.nc"
    generate_sample_configfile(out_path=configfile_path)

    water_varname_map = {
        "longitude": "lon",
        "latitude": "lat",
        "Z": "depth",
        "EVEL": "U",
        "NVEL": "V",
        "WVELMASS": "W",
    }

    out_dir = output_root / sourcefile_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # run the model!
    out_paths = run_advector_3D(
        output_directory=str(out_dir),
        sourcefile_path=str(sourcefile_path),
        configfile_path=str(configfile_path),
        u_water_path=str(data_root / "EVEL_2015*.nc"),
        v_water_path=str(data_root / "NVEL_2015*.nc"),
        w_water_path=str(data_root / "WVELMASS_2015*.nc"),
        water_varname_map=water_varname_map,
        u_wind_path=str(data_root / "uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(data_root / "vwnd.10m.gauss.2015.nc"),
        wind_varname_map={"uwnd": "U", "vwnd": "V", "level": "depth"},  # wind
        seawater_density_path=str(data_root / "RHO_2015.nc"),
        advection_start_date=datetime(year=2015, month=1, day=1, hour=12),
        timestep=timedelta(hours=1),
        num_timesteps=24 * 365,
        save_period=24,
        memory_utilization=0.4,
    )

    water_varname_map.pop("NVEL")
    water_varname_map.pop("WVELMASS")
    for out_path in out_paths:
        print("Animating trajectories...")
        animate_ocean_advection(out_path)
