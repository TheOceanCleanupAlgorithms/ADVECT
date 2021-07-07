"""
advect on ECCO surface currents
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr

examples_root = Path(__file__).parent
sys.path.append(str(examples_root.parent))
sys.path.append(str(examples_root.parent / "src"))

from helpers.generate_sourcefiles import generate_2D_sourcefile
from src.run_advector_2D import run_advector_2D
from src.plotting.plot_advection import animate_ocean_advection, plot_ocean_trajectories


if __name__ == "__main__":
    data_root = Path(input("Input path to example data directory: "))
    output_root = Path(input("Input path to directory for outputfiles: "))
    output_root.mkdir(exist_ok=True)

    ADVECTION_START = datetime(2015, 1, 1)
    ADVECTION_END = datetime(2015, 2, 1)

    sourcefile_path = output_root / "2D_uniform_source_2015.nc"
    generate_2D_sourcefile(
        num_particles=5000,
        release_date_range=(
            ADVECTION_START,
            ADVECTION_START + (ADVECTION_END - ADVECTION_START) / 2,
        ),
        out_path=sourcefile_path,
    )

    out_dir = output_root / sourcefile_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_currents(currents: xr.Dataset) -> xr.Dataset:
        return currents.rename(
            {
                "longitude": "lon",
                "latitude": "lat",
                "Z": "depth",
                "EVEL": "U",
                "NVEL": "V",
            }
        )

    def preprocess_wind(wind: xr.Dataset) -> xr.Dataset:
        return wind.rename({"uwnd": "U", "vwnd": "V", "level": "depth"})

    out_paths = run_advector_2D(
        output_directory=str(out_dir),
        sourcefile_path=str(sourcefile_path),
        u_water_path=str(data_root / "EVEL_2015_01*.nc"),
        v_water_path=str(data_root / "NVEL_2015_01*.nc"),
        water_preprocessor=preprocess_currents,
        u_wind_path=str(data_root / "uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(data_root / "vwnd.10m.gauss.2015.nc"),
        wind_preprocessor=preprocess_wind,
        windage_coeff=0.005,  # fraction of windspeed transferred to particle
        eddy_diffusivity=200,  # m^2 / s
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24 * (ADVECTION_END - ADVECTION_START).days,
        save_period=4,
    )

    for path in out_paths:
        print("Animating trajectories...")
        animate_ocean_advection(
            outputfile_path=path,
            save=False,
        )
        print("Plotting trajectories...")
        plot_ocean_trajectories(outputfile_path=path)
