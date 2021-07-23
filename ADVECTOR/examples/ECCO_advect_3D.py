"""
advect on ECCO currents
"""
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr

from ADVECTOR.examples.helpers.generate_sourcefiles import generate_3D_sourcefile
from ADVECTOR.examples.helpers.generate_configfile import generate_sample_configfile
from ADVECTOR.plotting.plot_advection import animate_ocean_advection
from ADVECTOR.run_advector_3D import run_advector_3D


def main():
    data_root = Path(input("Input path to example data directory: "))
    output_root = Path(input("Input path to directory for outputfiles: "))
    output_root.mkdir(exist_ok=True)

    ADVECTION_START = datetime(2015, 1, 1)
    ADVECTION_END = datetime(2015, 2, 1)

    # generate a sourcefile
    sourcefile_path = output_root / "3D_uniform_source_2015.nc"
    generate_3D_sourcefile(
        num_particles=5000,
        density_range=(800, 1000),
        radius_range=(1e-6, 1),
        corey_shape_factor_range=(0.15, 1),
        release_date_range=(
            ADVECTION_START,
            ADVECTION_START + (ADVECTION_END - ADVECTION_START) / 2,
        ),
        depth_range=(0, 0),
        out_path=sourcefile_path,
    )
    # generate a configfile
    configfile_path = output_root / "config.nc"
    generate_sample_configfile(out_path=configfile_path)

    def preprocess_currents(currents: xr.Dataset) -> xr.Dataset:
        return currents.rename(
            {
                "longitude": "lon",
                "latitude": "lat",
                "Z": "depth",
                "EVEL": "U",
                "NVEL": "V",
                "WVELMASS": "W",
            }
        )

    def preprocess_wind(wind: xr.Dataset) -> xr.Dataset:
        return wind.rename({"uwnd": "U", "vwnd": "V", "level": "depth"})

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
        water_preprocessor=preprocess_currents,
        u_wind_path=str(data_root / "uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(data_root / "vwnd.10m.gauss.2015.nc"),
        wind_preprocessor=preprocess_wind,
        seawater_density_path=str(data_root / "RHO_2015.nc"),
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24 * (ADVECTION_END - ADVECTION_START).days,
        save_period=4,
        memory_utilization=0.4,
        overwrite_existing_files=True,
    )

    for out_path in out_paths:
        print("Animating trajectories...")
        animate_ocean_advection(out_path)


if __name__ == "__main__":
    main()
