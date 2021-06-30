"""
This is the ADVECTOR 2D entry-point.
To use, add this repo's "src" directory to the python path, import this file, and execute.  E.g.
    import sys
    sys.path.append("<path_to_repo>/src")
    from run_advector_2D import run_advector_2D
    run_advector_2D(...)
See examples/ECCO_advect_2D.py for an example usage.
See docstring below for descriptions of arguments.
See src/forcing_data_specifications.md for detailed description of forcing data requirements.
See src/sourcefile_specifications.md for detailed description of sourcefile requirements.
See src/outputfile_specifications.md for detailed description of the outputfile created by this program.
"""

import datetime
from pathlib import Path
from typing import Tuple

from dask.diagnostics import ProgressBar

from drivers.chunked_kernel_driver import execute_chunked_kernel_computation
from enums.advection_scheme import AdvectionScheme
from enums.forcings import Forcing
from io_tools.OutputWriter import OutputWriter2D
from io_tools.open_sourcefiles import open_2d_sourcefiles
from io_tools.open_vectorfiles import *
from kernel_wrappers.Kernel2D import Kernel2D, Kernel2DConfig


def run_advector_2D(
    sourcefile_path: str,
    output_directory: str,
    u_water_path: str,
    v_water_path: str,
    advection_start_date: datetime.datetime,
    timestep: datetime.timedelta,
    num_timesteps: int,
    eddy_diffusivity: float = 0,
    advection_scheme: str = "taylor2",
    save_period: int = 1,
    sourcefile_varname_map: Optional[dict] = None,
    water_varname_map: Optional[dict] = None,
    opencl_device: Tuple[int, ...] = None,
    memory_utilization: float = 0.4,
    u_wind_path: Optional[str] = None,
    v_wind_path: Optional[str] = None,
    wind_varname_map: Optional[dict] = None,
    windage_coeff: Optional[float] = None,
    show_progress_bar: bool = True,
) -> List[str]:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.
        Can be a wildcard path as long as the individual sourcefiles can be properly concatenated along particle axis.
        See data_specifications.md for data requirements.
    :param output_directory: directory which will be populated with the outfiles.
        Existing files in this directory may be overwritten.
        See data_specifications.md for outputfile format details.
    :param u_water_path: wildcard path to the zonal current files.
        See data_specifications.md for data requirements.
    :param v_water_path: wildcard path to the meridional current files; see 'u_water_path'.
    :param advection_start_date: python datetime object denoting the start of the advection timeseries.
        Any particles which are scheduled to be released prior to this date will be released at this date.
    :param timestep: python timedelta object denoting the duration of each advection timestep.
    :param num_timesteps: length of the advection timeseries.
    :param eddy_diffusivity: (m^2 / s) controls the scale of each particle's random walk.  0 (default) has no effect.
        Note: since eddy diffusivity parameterizes ocean mechanics at smaller scales than the current files resolve,
            the value chosen should reflect the resolution of the current files.  Further, though eddy diffusivity in
            the real ocean varies widely in space and time, ADVECTOR uses one value everywhere, and the value should be
            selected with this in mind.
    :param advection_scheme: one of {"taylor2", "eulerian"}.
        "taylor2" is a second-order advection scheme as described in Black/Gay 1990 which improves adherence to circular
            streamlines compared to a first-order scheme.  This is the default.
        "eulerian" is the forward Euler method.
    :param save_period: controls how often to write output: particle state will be saved every {save_period} timesteps.
        For example, with timestep=one hour, and save_period=24, the particle state will be saved once per day.
    :param sourcefile_varname_map: mapping from names in sourcefile to standard names, as defined in
        data_specifications.md.  E.g. {"longitude": "lon", "particle_release_time": "release_date", ...}
    :param water_varname_map: mapping from names in current files to standard names.  See 'sourcefile_varname_map'.
    :param opencl_device: specifies hardware for computation.  If None (default), the user will receive a series of
        prompts which guides them through selecting a compute device.  To bypass this prompt, you can encode your
        answers to each of the prompts in a tuple, e.g. (0, 2).
    :param memory_utilization: this defines what percentage of the opencl device memory will be assumed usable.
        If you are using the CPU as your opencl device, don't set this above .5, because the CPU memory is shared by the
        host code (i.e. python-side), among other things.
        If you are using a dedicated compute device, this can be set close to 1.
        For example, on a dedicated compute device (e.g. GPU) with 2GB memory, setting memory_utilization = .95
            will allow the program to send a maximum of 1.9GB to the GPU at once.
        In general, set this as high as you can without running out of memory.
    :param u_wind_path: wildcard path to zonal 10-meter wind files; see 'u_water_path'.
        Wind is optional.  Simply omit this argument in order to disable drift due to wind.
    :param v_wind_path: wildcard path to meridional 10-meter wind files; see 'u_wind_path'.
    :param wind_varname_map mapping from names in wind file to standard names.  See 'sourcefile_varname_map'.
    :param windage_coeff: fraction of wind speed that is transferred to particle.
        If u_wind_path is specified, i.e., wind is enabled, this value must be specified.
        Note: this value has a profound impact on results.
    :param show_progress_bar: whether to show progress bars for dask operations
    :return: list of paths to the outputfiles
    """
    if show_progress_bar:
        ProgressBar(minimum=1).register()
    arguments = locals()
    try:
        scheme_enum = AdvectionScheme[advection_scheme]
    except KeyError:
        raise ValueError(
            f"Invalid argument advection_scheme; must be one of "
            f"{set(scheme.name for scheme in AdvectionScheme)}."
        )

    print("---INITIALIZING DATASETS---")
    print("Opening Sourcefiles...")
    p0 = open_2d_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=sourcefile_varname_map,
    )

    forcing_data = {}
    print("Initializing Ocean Current...")
    forcing_data[Forcing.current] = open_2d_currents(
        u_path=u_water_path, v_path=v_water_path, variable_mapping=water_varname_map
    )

    if u_wind_path is not None and v_wind_path is not None:
        print("Initializing Wind...")
        forcing_data[Forcing.wind] = open_wind(
            u_path=u_wind_path, v_path=v_wind_path, variable_mapping=wind_varname_map
        )

    output_writer = OutputWriter2D(
        out_dir=Path(output_directory),
        basename="ADVECTOR_2D_output",
        sourcefile=p0,
        forcing_data=forcing_data,
        api_entry="src/run_advector_2D.py::run_advector_2D",
        api_arguments=arguments,
    )

    print("---COMMENCING ADVECTION---")
    out_paths = execute_chunked_kernel_computation(
        forcing_data=forcing_data,
        kernel_cls=Kernel2D,
        kernel_config=Kernel2DConfig(
            advection_scheme=scheme_enum,
            windage_coefficient=windage_coeff,
            eddy_diffusivity=eddy_diffusivity,
        ),
        output_writer=output_writer,
        p0=p0,
        start_time=advection_start_date,
        dt=timestep,
        num_timesteps=num_timesteps,
        save_every=save_period,
        platform_and_device=opencl_device,
        memory_utilization=memory_utilization,
    )

    return [str(p) for p in out_paths]
