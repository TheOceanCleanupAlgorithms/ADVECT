"""
the master advection runner.  Takes as input a path to a particle sourcefile, and a wildcard path for each current
variable.  Outputs the advection results to a particle sourcefile.
"""
import datetime
import json
import click
from pathlib import Path
from typing import Optional, Tuple

from drivers.opencl_driver_2D import openCL_advect
from kernel_wrappers.Kernel2D import AdvectionScheme
from io_tools.open_vectorfiles import open_netcdf_vectorfield
from io_tools.open_sourcefiles import SourceFileType, open_sourcefiles
from dateutil import parser

DEFAULT_EDDY_DIFFUSIVITY = 0
DEFAULT_WINDAGE_COEFF = 0
DEFAULT_SAVE_PERIOD = 1


def run_advector(
    sourcefile_path: str,
    outputfile_path: str,
    u_water_path: str,
    v_water_path: str,
    advection_start: str,
    timestep_seconds: float,
    num_timesteps: int,
    advection_scheme: AdvectionScheme,
    eddy_diffusivity: float = DEFAULT_EDDY_DIFFUSIVITY,
    windage_coeff: float = DEFAULT_WINDAGE_COEFF,
    save_period: int = DEFAULT_SAVE_PERIOD,
    source_file_type: SourceFileType = SourceFileType.new_source_files,
    sourcefile_varname_map: dict = None,
    currents_varname_map: dict = None,
    platform_and_device: Tuple[int, ...] = None,
    verbose: bool = False,
    memory_utilization: float = 0.5,
    u_wind_path: Optional[str] = None,
    v_wind_path: Optional[str] = None,
    windfile_varname_map: dict = None,
) -> str:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param outputfile_path: path which will be populated with the outfile.
    :param u_water_path: wildcard path to the zonal current files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_water_path: wildcard path to the meridional current files.  See u_path for more details.
    :param advection_start: ISO 8601 datetime string.
    :param timestep_seconds: duration of each timestep in seconds
    :param num_timesteps: number of timesteps
    :param advection_scheme: which numerical advection scheme to use
    :param eddy_diffusivity: (m^2 / s) constant controlling the scale of each particle's random walk; model dependent
    :param windage_coeff: float in [0, 1], fraction of wind speed that is transferred to particle
    :param save_period: how often to write output.  Particle state will be saved every {save_period} timesteps.
    :param source_file_type: enum of what format source file is input
    :param sourcefile_varname_map: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('id', 'lat', 'lon', 'release_date')
    :param currents_varname_map: mapping from names in current file to advector standard variable names
            advector standard names: ('U', 'V', 'W', 'lat', 'lon', 'time', 'depth')
    :param platform_and_device: [index of opencl platform, index of opencl device] to specify hardware for computation
    :param verbose: whether to print out a bunch of extra stuff
    :param memory_utilization: this defines what percentage of the device memory advector will use for opencl buffers.
                                if using a separate, dedicated opencl device (e.g. GPU) try something close to 1.
                                if using the main CPU, try something close to .5.
    :param u_wind_path: wildcard path to zonal surface wind files.  Assumes sorting paths by name == sorting paths in time
    :param v_wind_path: wildcard path to meridional surface wind files.
    :param windfile_varname_map mapping from names in current file to advector standard variable names
            advector standard names: ('U', 'V', 'lat', 'lon', 'time')
    :return: absolute path to the particle outputfile
    """
    if sourcefile_varname_map is None:
        sourcefile_varname_map = {}
    p0 = open_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=sourcefile_varname_map,
        source_file_type=source_file_type,
    )
    currents = open_netcdf_vectorfield(
        u_path=u_water_path, v_path=v_water_path, variable_mapping=currents_varname_map
    )
    if u_wind_path is not None:
        wind = open_netcdf_vectorfield(
            u_path=u_wind_path, v_path=v_wind_path, variable_mapping=windfile_varname_map
        )
    else:
        wind = None

    start_date = parser.isoparse(advection_start)  # python datetime.datetime
    dt = datetime.timedelta(seconds=timestep_seconds)

    openCL_advect(
        current=currents,
        wind=wind,
        out_path=Path(outputfile_path),
        p0=p0,
        start_time=start_date,
        dt=dt,
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


@click.command()
@click.option("--source", "sourcefile_path", required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              )
@click.option("--output", "outputfile_path", required=True,
              type=click.Path(exists=False, dir_okay=False, readable=True),
              )
@click.option("--u", "u_path", required=True, type=click.STRING)
@click.option("--v", "v_path", required=True, type=click.STRING)
@click.option("--start", "advection_start", required=True, type=click.STRING)
@click.option("--dt", "timestep_seconds", required=True, type=click.FLOAT)
@click.option("--nt", "num_timesteps", required=True, type=click.INT)
@click.option('--scheme', "advection_scheme", required=True,
              type=click.Choice([s.name for s in AdvectionScheme], case_sensitive=True))
@click.option('--eddy_diff', "eddy_diffusivity", required=False, default=DEFAULT_EDDY_DIFFUSIVITY)
@click.option('--windage', "windage_coeff", required=False, default=DEFAULT_WINDAGE_COEFF)
@click.option("--save_period", "save_period", required=False, default=DEFAULT_SAVE_PERIOD)
@click.option("--source_type", "source_file_type", required=False, default=SourceFileType.new_source_files.name,
              type=click.Choice([t.name for t in SourceFileType]))
@click.option("--source_name_map", "sourcefile_varname_map", required=False, type=click.STRING)
@click.option("--currents_name_map", "sourcefile_varname_map", required=False, type=click.STRING)
@click.option("--cl_platform", "cl_platform", required=False, type=click.INT)
@click.option("--cl_device", "cl_device", required=False, type=click.INT)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--mem_util", "memory_utilization", required=False, default=0.5)
def run_advector_CLI(
    advection_scheme: str,
    sourcefile_varname_map: str = None,
    currents_varname_map: str = None,
    cl_platform: int = None,
    cl_device: int = None,
    **kwargs,
):
    platform_and_device = None if cl_platform is None or cl_device is None else (cl_platform, cl_device)
    if sourcefile_varname_map:
        sourcefile_varname_map = json.loads(sourcefile_varname_map)
    if currents_varname_map:
        currents_varname_map = json.loads(currents_varname_map)

    run_advector(advection_scheme=AdvectionScheme[advection_scheme],
                 sourcefile_varname_map=sourcefile_varname_map,
                 currents_varname_map=currents_varname_map,
                 platform_and_device=platform_and_device,
                 **kwargs)


if __name__ == '__main__':
    run_advector_CLI()
