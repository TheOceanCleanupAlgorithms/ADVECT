"""
advect on ECCO surface currents
"""
from datetime import datetime, timedelta
from pathlib import Path

from sourcefiles.generate_sourcefiles import generate_uniform_2D_sourcefile
from ADVECTOR.run_advector_2D import run_advector_2D
from ADVECTOR.plotting.plot_advection import (
    animate_ocean_advection,
    plot_ocean_trajectories,
)


if __name__ == "__main__":
    data_root = Path(input("Input path to example data directory: "))
    output_root = Path(input("Input path to directory for outputfiles: "))
    output_root.mkdir(parents=True, exist_ok=True)

    sourcefile_path = output_root / "2D_uniform_source_2015.nc"
    generate_uniform_2D_sourcefile(
        out_path=sourcefile_path,
    )

    ADVECTION_START = datetime(2015, 1, 1)
    ADVECTION_END = datetime(2016, 1, 1)
    out_paths = run_advector_2D(
        output_directory=str(output_root / f"/ECCO_2015_2D/{sourcefile_path.stem}"),
        sourcefile_path=str(sourcefile_path),
        u_water_path=str(data_root / "U_2015*.nc"),
        v_water_path=str(data_root / "V_2015*.nc"),
        u_wind_path=str(data_root / "uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(data_root / "vwnd.10m.gauss.2015.nc"),
        wind_varname_map={"uwnd": "U", "vwnd": "V", "level": "depth"},  # wind
        windage_coeff=0.005,  # fraction of windspeed transferred to particle
        eddy_diffusivity=200,  # m^2 / s
        advection_start_date=ADVECTION_START,
        timestep=timedelta(hours=1),
        num_timesteps=24 * (ADVECTION_END - ADVECTION_START).days,
        save_period=24,
    )

    for path in out_paths:
        print("Animating trajectories...")
        animate_ocean_advection(
            outputfile_path=path,
            save=False,
            current_path=str(data_root / "U_2015-01-01.nc"),
        )
        print("Plotting trajectories...")
        plot_ocean_trajectories(path, str(data_root / "U_2015-01-01.nc"))
