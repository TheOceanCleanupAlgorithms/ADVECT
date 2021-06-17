"""
advect on ECCO currents
"""
from datetime import datetime, timedelta
from pathlib import Path


from sourcefiles.generate_sourcefiles import generate_uniform_3D_sourcefile
from configfiles.generate_configfile import generate_sample_configfile
from ADVECTOR.plotting.plot_advection import animate_ocean_advection
from ADVECTOR.run_advector_3D import run_advector_3D


if __name__ == "__main__":
    data_root = Path(input("Input path to example data directory: "))
    output_root = Path(input("Input path to directory for outputfiles: "))
    output_root.mkdir(parents=True, exist_ok=True)
    # generate a sourcefile
    sourcefile_path = output_root / "3D_uniform_source_2015.nc"
    generate_uniform_3D_sourcefile(out_path=sourcefile_path)

    # generate a configfile
    configfile_path = output_root / "config.nc"
    generate_sample_configfile(out_path=configfile_path)

    # run the model!
    out_paths = run_advector_3D(
        output_directory=str(
            output_root / f"ECCO_2015_3D/{Path(sourcefile_path).stem}/"
        ),
        sourcefile_path=str(sourcefile_path),
        configfile_path=str(configfile_path),
        u_water_path=str(data_root / "U_2015*.nc"),
        v_water_path=str(data_root / "V_2015*.nc"),
        w_water_path=str(data_root / "W_2015*.nc"),
        u_wind_path=str(data_root / "uwnd.10m.gauss.2015.nc"),
        v_wind_path=str(data_root / "vwnd.10m.gauss.2015.nc"),
        wind_varname_map={"uwnd": "U", "vwnd": "V", "level": "depth"},  # wind
        seawater_density_path=str(data_root / "RHO_2015.nc"),
        advection_start_date=datetime(year=2015, month=1, day=1, hour=12),
        timestep=timedelta(hours=1),
        num_timesteps=24 * 365,
        save_period=24,
    )

    for out_path in out_paths:
        print("Animating trajectories...")
        animate_ocean_advection(
            out_path,
            current_path=str(data_root / "U_2015-01-01.nc"),
        )
