"""
the master advection runner.  Takes as input a path to a particle sourcefile, and a wildcard path for each current
variable.  Outputs the advection results to a particle sourcefile.
"""
from tools.open_sourcefile import open_sourcefile
from dateutil import parser


def run_advector(
    sourcefile_path: str,
    advection_start: str,
    timestep_seconds: float,
    num_timesteps: int,
    save_period: int,
) -> str:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.  Absolute path safest, use relative paths with caution.
    :param advection_start: ISO 8601 datetime string.
    :param timestep_seconds: duration of each timestep in seconds
    :param num_timesteps: number of timesteps
    :param save_period: how often to write output.  Particle state will be saved every {save_period} timesteps.
    :return: absolute path to the particle outputfile
    """
    p0 = open_sourcefile(sourcefile_path)
    start_date = parser.isoparse(advection_start)  # python datetime.datetime

    return ""
