"""
This is the ADVECTOR entry-point.
To use, create a python script within this repo, import this file, and execute.  E.g.
    from run_advector import run_advector
    run_advector(...)
See examples/HYCOM_advect_2d.py for an example usage.
See function docstring below for detailed descriptions of all arguments.
See src/data_specifications.md for detailed description of data format requirements.
"""

import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple, List

from drivers.opencl_driver_2D import openCL_advect
from kernel_wrappers.Kernel2D import AdvectionScheme
from io_tools.open_sourcefiles import SourceFileFormat, open_sourcefiles
from io_tools.open_vectorfiles import open_netcdf_vectorfield, empty_vectorfield


def run_advector(
    sourcefile_path: str,
    output_directory: str,
    u_water_path: str,
    v_water_path: str,
    advection_start_date: datetime.datetime,
    timestep: datetime.timedelta,
    num_timesteps: int,
    eddy_diffusivity: float = 0,
    advection_scheme: str = 'taylor2',
    save_period: int = 1,
    sourcefile_format: str = 'advector',
    sourcefile_varname_map: Optional[dict] = None,
    opencl_device: Tuple[int, ...] = None,
    memory_utilization: float = 0.5,
    u_wind_path: Optional[str] = None,
    v_wind_path: Optional[str] = None,
    windage_coeff: Optional[float] = None,
    verbose: bool = False,
    water_preprocessor: Optional[Callable] = None,
    wind_preprocessor: Optional[Callable] = None,
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
    :param sourcefile_format: one of {"advector", "trashtracker"}.  See data_specifications.md for more details.
    :param sourcefile_varname_map: mapping from names in sourcefile to standard names, as defined in
        data_specifications.md.  E.g. {"longitude": "lon", "particle_release_time": "release_date", ...}
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
    :param u_wind_path: wildcard path to zonal surface wind files; see 'u_water_path'.
        Wind is optional.  Simply omit this argument in order to disable drift due to wind.
    :param v_wind_path: wildcard path to meridional surface wind files; see 'u_wind_path'.
    :param windage_coeff: fraction of wind speed that is transferred to particle.
        If u_wind_path is specified, i.e., wind is enabled, this value must be specified.
        Note: this value has a profound impact on results.
    :param verbose: whether to print detailed information about kernel execution.
    :param water_preprocessor: func to call on the xarray water dataset to perform operation before loading in advector, such as renaming variables.
    :param wind_preprocessor: func to call on the xarray wind dataset to perform operation before loading in advector, such as renaming variables.
    """
    try:
        scheme_enum = AdvectionScheme[advection_scheme]
    except KeyError:
        raise ValueError(f"Invalid argument advection_scheme; must be one of "
                         f"{set(scheme.name for scheme in AdvectionScheme)}.")
    try:
        sourcefile_format_enum = SourceFileFormat[sourcefile_format]
    except KeyError:
        raise ValueError(f"Invalid argument sourcefile_format; must be one of "
                         f"{set(fmt.name for fmt in SourceFileFormat)}.")

    p0 = open_sourcefiles(
        sourcefile_path=sourcefile_path,
        variable_mapping=sourcefile_varname_map,
        source_file_type=sourcefile_format_enum,
    )
    currents = open_netcdf_vectorfield(
        u_path=u_water_path, v_path=v_water_path, preprocessor=water_preprocessor
    )

    if u_wind_path is not None and v_wind_path is not None:
        assert windage_coeff is not None, "Wind data must be accompanied by windage coefficient."
        wind = open_netcdf_vectorfield(
            u_path=u_wind_path, v_path=v_wind_path, preprocessor=wind_preprocessor
        )
    else:
        wind = empty_vectorfield()
        windage_coeff = None  # this is how we flag windage=off

    out_paths = openCL_advect(
        current=currents,
        wind=wind,
        out_dir=Path(output_directory),
        p0=p0,
        start_time=advection_start_date,
        dt=timestep,
        num_timesteps=num_timesteps,
        save_every=save_period,
        advection_scheme=scheme_enum,
        eddy_diffusivity=eddy_diffusivity,
        windage_coeff=windage_coeff,
        platform_and_device=opencl_device,
        verbose=verbose,
        memory_utilization=memory_utilization,
    )

    return [str(p) for p in out_paths]
