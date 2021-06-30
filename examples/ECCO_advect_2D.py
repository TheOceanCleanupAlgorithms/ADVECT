"""
advect on ECCO surface currents
"""
from datetime import datetime, timedelta
from pathlib import Path

from helpers.generate_sourcefiles import generate_2D_sourcefile
from ADVECTOR.run_advector_2D import run_advector_2D
from ADVECTOR.plotting.plot_advection import (
    animate_ocean_advection,
    plot_ocean_trajectories,
)


if __name__ == "__main__":
    data_root = Path(
        "/Users/dklink/data_science/metocean_data/ADVECTOR_sample_data"
    )  # input("Input path to example data directory: "))
    output_root = Path(
        "/Users/dklink/Desktop/outputfiles"
    )  # input("Input path to directory for outputfiles: "))
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

    water_varname_map = {
        "longitude": "lon",
        "latitude": "lat",
        "Z": "depth",
        "EVEL": "U",
        "NVEL": "V",
    }
    out_paths = run_advector_2D(
        output_directory=str(out_dir),
        sourcefile_path=str(sourcefile_path),
        u_water_path=str(data_root / "EVEL_2015_01*.nc"),
        v_water_path=str(data_root / "NVEL_2015_01*.nc"),
        water_varname_map=water_varname_map,
        u_wind_path=str(data_root / "uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(data_root / "vwnd.10m.gauss.2015.nc"),
        wind_varname_map={"uwnd": "U", "vwnd": "V", "level": "depth"},  # wind
        windage_coeff=0.005,  # fraction of windspeed transferred to particle
        eddy_diffusivity=200,  # m^2 / s
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24 * (ADVECTION_END - ADVECTION_START).days,
        save_period=4,
    )

    water_varname_map.pop("NVEL")
    for path in out_paths:
        print("Animating trajectories...")
        animate_ocean_advection(
            outputfile_path=path,
            save=False,
            current_path=str(data_root / "EVEL_2015_01_01.nc"),
            current_varname_map=water_varname_map,
        )
        print("Plotting trajectories...")
        plot_ocean_trajectories(
            outputfile_path=path,
            current_path=str(data_root / "EVEL_2015_01_01.nc"),
            current_varname_map=water_varname_map,
        )
