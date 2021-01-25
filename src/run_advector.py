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
from typing import Tuple, List

from drivers.opencl_driver_3D import openCL_advect
from io_tools.OutputWriter import OutputWriter
from io_tools.open_configfiles import load_eddy_diffusivity, load_density_profile
from kernel_wrappers.Kernel3D import AdvectionScheme
from io_tools.open_sourcefiles import SourceFileFormat, open_sourcefiles
from io_tools.open_vectorfiles import open_2D_vectorfield, empty_2D_vectorfield, open_3D_vectorfield


def run_advector(
    sourcefile_path: str,
    configfile_path: str,
    output_directory: str,
    u_water_path: str,
    v_water_path: str,
    w_water_path: str,
    advection_start_date: datetime.datetime,
    timestep: datetime.timedelta,
    num_timesteps: int,
    advection_scheme: str = 'taylor2',
    save_period: int = 1,
    sourcefile_format: str = 'advector',
    sourcefile_varname_map: dict = None,
    water_varname_map: dict = None,
    opencl_device: Tuple[int, ...] = None,
    memory_utilization: float = 0.5,
    u_wind_path: str = None,
    v_wind_path: str = None,
    wind_varname_map: dict = None,
    windage_multiplier: float = 1,
    verbose: bool = False,
) -> List[str]:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.
        Can be a wildcard path as long as the individual sourcefiles can be properly concatenated along particle axis.
        See data_specifications.md for data requirements.
    :param configfile_path: path to the configfile netcdf file.
        See config_specifications.md for details
    :param output_directory: directory which will be populated with the outfiles.
        Existing files in this directory may be overwritten.
        See data_specifications.md for outputfile format details.
    :param u_water_path: wildcard path to the zonal current files.
        See data_specifications.md for data requirements.
    :param v_water_path: wildcard path to the meridional current files; see 'u_water_path'.
    :param w_water_path: wildcard path to the vertical current files; see 'u_water_path'.
    :param advection_start_date: python datetime object denoting the start of the advection timeseries.
        Any particles which are scheduled to be released prior to this date will be released at this date.
    :param timestep: python timedelta object denoting the duration of each advection timestep.
    :param num_timesteps: length of the advection timeseries.
    :param advection_scheme: one of {"taylor2", "eulerian"}.
        "taylor2" is a second-order advection scheme as described in Black/Gay 1990 which improves adherence to circular
            streamlines compared to a first-order scheme.  This is the default.
        "eulerian" is the forward Euler method.
    :param save_period: controls how often to write output: particle state will be saved every {save_period} timesteps.
        For example, with timestep=one hour, and save_period=24, the particle state will be saved once per day.
    :param sourcefile_format: one of {"advector", "trashtracker"}.  See data_specifications.md for more details.
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
    :param windage_multiplier: multiplies the default windage, which is based on emerged area.
    :param verbose: whether to print detailed information about kernel execution.
    :return: list of paths to the outputfiles
    """
    arguments = locals()
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

    currents = open_3D_vectorfield(
        u_path=u_water_path, v_path=v_water_path, w_path=w_water_path, variable_mapping=water_varname_map
    )

    if u_wind_path is not None and v_wind_path is not None:
        assert windage_multiplier is not None, "Wind data must be accompanied by windage coefficient."
        wind = open_2D_vectorfield(
            u_path=u_wind_path, v_path=v_wind_path, variable_mapping=wind_varname_map
        )
    else:
        wind = empty_2D_vectorfield()
        windage_multiplier = None  # this is how we flag windage=off

    eddy_diffusivity = load_eddy_diffusivity(configfile_path=configfile_path)
    density_profile = load_density_profile(configfile_path=configfile_path)

    output_writer = OutputWriter(
        out_dir=Path(output_directory),
        configfile_path=configfile_path,
        sourcefile_path=sourcefile_path,
        currents=currents,
        wind=wind if windage_multiplier is not None else None,
        arguments_to_run_advector=arguments,
    )

    out_paths = openCL_advect(
        current=currents,
        wind=wind,
        output_writer=output_writer,
        p0=p0,
        start_time=advection_start_date,
        dt=timestep,
        num_timesteps=num_timesteps,
        save_every=save_period,
        advection_scheme=scheme_enum,
        eddy_diffusivity=eddy_diffusivity,
        density_profile=density_profile,
        windage_multiplier=windage_multiplier,
        platform_and_device=opencl_device,
        verbose=verbose,
        memory_utilization=memory_utilization,
    )

    return [str(p) for p in out_paths]
