"""
Top-level entry point.  Must be called from a python script; see examples/HYCOM_advect_2d.py for example usage
"""
import datetime
from pathlib import Path
from typing import Optional, Tuple

from drivers.opencl_driver_2D import openCL_advect
from kernel_wrappers.Kernel2D import AdvectionScheme
from io_tools.open_sourcefiles import SourceFileType, open_sourcefiles
from io_tools.open_vectorfiles import open_netcdf_vectorfield, empty_vectorfield, open_3D_vectorfield

DEFAULT_EDDY_DIFFUSIVITY = 0
DEFAULT_WINDAGE_COEFF = 0
DEFAULT_SAVE_PERIOD = 1


def run_advector(
    sourcefile_path: str,
    outputfile_path: str,
    u_water_path: str,
    v_water_path: str,
    advection_start_date: datetime.datetime,
    timestep: datetime.timedelta,
    num_timesteps: int,
    eddy_diffusivity: float,
    advection_scheme: AdvectionScheme = AdvectionScheme.eulerian3d,
    save_period: int = 1,
    source_file_type: SourceFileType = SourceFileType.advector,
    sourcefile_varname_map: dict = None,
    currents_varname_map: Optional[dict] = None,
    platform_and_device: Tuple[int, ...] = None,
    verbose: bool = False,
    memory_utilization: float = 0.5,
    u_wind_path: Optional[str] = None,
    v_wind_path: Optional[str] = None,
    windfile_varname_map: Optional[dict] = None,
    windage_coeff: Optional[float] = None,
    w_water_path: Optional[str] = None,
) -> str:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param outputfile_path: path which will be populated with the outfile.
    :param u_water_path: wildcard path to the zonal current files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_water_path: wildcard path to the meridional current files.  See u_path for more details.
    :param w_water_path: wildcard path to the vertical current files.  See u_path for more details.
    :param advection_start_date: date the advection clock starts.
    :param timestep: duration of each advection timestep
    :param num_timesteps: number of timesteps
    :param eddy_diffusivity: (m^2 / s) constant controlling the scale of each particle's random walk; model dependent
    :param advection_scheme: which numerical advection scheme to use
    :param save_period: how often to write output.  Particle state will be saved every {save_period} timesteps.
    :param source_file_type: enum of what format source file is input
    :param sourcefile_varname_map: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('p_id', 'lat', 'lon', 'release_date')
    :param currents_varname_map: mapping from names in current file to advector standard variable names
            advector standard names: ('U', 'V', 'W', 'lat', 'lon', 'time', 'depth')
    :param platform_and_device: [index of opencl platform, index of opencl device] to specify hardware for computation
    :param verbose: whether to print out a bunch of extra stuff
    :param memory_utilization: this defines what percentage of the device memory will be used for opencl buffers.
                                if using a separate, dedicated opencl device (e.g. GPU) try something close to 1.
                                if using the main CPU, try something close to .5.
    :param u_wind_path: wildcard path to zonal surface wind files.
    :param v_wind_path: wildcard path to meridional surface wind files.
    :param windfile_varname_map mapping from names in current file to advector standard variable names
            advector standard names: ('U', 'V', 'lat', 'lon', 'time')
    :param windage_coeff: float in [0, 1], fraction of wind speed that is transferred to particle
    :return: absolute path to the particle outputfile
    """
    p0 = open_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=sourcefile_varname_map,
        source_file_type=source_file_type,
    )
    if w_water_path is not None:
        currents = open_3D_vectorfield(
            u_path=u_water_path, v_path=v_water_path, w_path=w_water_path, variable_mapping=currents_varname_map
        )
    else:
        currents = open_netcdf_vectorfield(
            u_path=u_water_path, v_path=v_water_path, variable_mapping=currents_varname_map
        )

    if u_wind_path is not None and v_wind_path is not None:
        assert windage_coeff is not None, "Wind data must be accompanied by windage coefficient."
        wind = open_netcdf_vectorfield(
            u_path=u_wind_path, v_path=v_wind_path, variable_mapping=windfile_varname_map
        )
    else:
        wind = empty_vectorfield()
        windage_coeff = None  # this is how we flag windage=off

    openCL_advect(
        current=currents,
        wind=wind,
        out_path=Path(outputfile_path),
        p0=p0,
        start_time=advection_start_date,
        dt=timestep,
        num_timesteps=num_timesteps,
        save_every=save_period,
        advection_scheme=advection_scheme,
        eddy_diffusivity=eddy_diffusivity,
        windage_coeff=windage_coeff,
        platform_and_device=platform_and_device,
        verbose=verbose,
        memory_utilization=memory_utilization,
    )

    return outputfile_path
