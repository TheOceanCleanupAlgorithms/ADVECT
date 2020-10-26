"""
the master advection runner.  Takes as input a path to a particle sourcefile, and a wildcard path for each current
variable.  Outputs the advection results to a particle sourcefile.
"""
import datetime
import os
from typing import Tuple

from drivers.opencl_driver_2D import openCL_advect
from kernel_wrappers.Kernel2D import AdvectionScheme
from tools.open_currentfiles import open_currentfiles
from tools.open_sourcefile import open_sourcefile
from dateutil import parser


def run_advector(
    output_dir: str,
    sourcefile_path: str,
    u_path: str,
    v_path: str,
    advection_start: str,
    timestep_seconds: float,
    num_timesteps: int,
    save_period: int,
    advection_scheme: AdvectionScheme,
    sourcefile_varname_map: dict = None,
    currents_varname_map: dict = None,
    platform_and_device: Tuple[int, int] = None,
    verbose: bool = None,
) -> str:
    """
    :param output_dir: path to the directory where the outputfile will be written
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param u_path: wildcard path to the zonal current files.  Fed to glob.glob.  Assumes sorting paths by name == sorting paths in time
    :param v_path: wildcard path to the zonal current files.  See u_path for more details.
    :param advection_start: ISO 8601 datetime string.
    :param timestep_seconds: duration of each timestep in seconds
    :param num_timesteps: number of timesteps
    :param save_period: how often to write output.  Particle state will be saved every {save_period} timesteps.
    :param advection_scheme: which numerical advection scheme to use
    :param sourcefile_varname_map: mapping from names in sourcefile to advector standard variable names
            advector standard names: ('id', 'lat', 'lon', 'release_date')
    :param currents_varname_map: mapping from names in current file to advector standard variable names
            advector standard names: ('U', 'V', 'W', 'lat', 'lon', 'time', 'depth')
    :param platform_and_device: [index of opencl platform, index of opencl device] to specify hardware for computation
    :param verbose: whether to print out a bunch of extra stuff
    :return: absolute path to the particle outputfile
    """
    if sourcefile_varname_map is None:
        sourcefile_varname_map = {}
    p0 = open_sourcefile(
        sourcefile_path=sourcefile_path, variable_mapping=sourcefile_varname_map
    )
    currents = open_currentfiles(
        u_path=u_path, v_path=v_path, variable_mapping=currents_varname_map
    )

    start_date = parser.isoparse(advection_start)  # python datetime.datetime
    dt = datetime.timedelta(seconds=timestep_seconds)
    P, buf_time, kernel_time = openCL_advect(
        field=currents,
        p0=p0,
        start_time=start_date,
        dt=dt,
        num_timesteps=num_timesteps,
        save_every=save_period,
        advection_scheme=advection_scheme,
        platform_and_device=platform_and_device,
        verbose=verbose,
    )
    out_path = os.path.join(
        output_dir, f"{datetime.datetime.utcnow().timestamp()}_advection.nc"
    )
    P.to_netcdf(out_path)

    return out_path
