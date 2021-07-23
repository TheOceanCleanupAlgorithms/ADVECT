"""
This is the ADVECTOR 3D entry-point.
To use, add this repo's "src" directory to the python path, import this file, and execute.  E.g.
    import sys
    sys.path.append("<path_to_repo>/src")
    from run_advector_3D import run_advector_3D
    run_advector_3D(...)
See examples/ECCO_advect_3D.py for an example usage.
See docstring below for descriptions of arguments.
See src/forcing_data_specifications.md for detailed description of forcing data requirements.
See src/sourcefile_specifications.md for detailed description of sourcefile requirements.
See src/configfile_specifications.md for detailed description of configfile requirements.
See src/outputfile_specifications.md for detailed description of the outputfile created by this program.
"""

import datetime
from pathlib import Path
from typing import Tuple, Optional, Union, Callable, List

import xarray as xr
from dask.diagnostics import ProgressBar

from drivers.chunked_kernel_driver import execute_chunked_kernel_computation
from enums.advection_scheme import AdvectionScheme
from enums.forcings import Forcing
from io_tools.OutputWriter import OutputWriter3D
from io_tools.open_configfiles import unpack_configfile
from io_tools.open_sourcefiles import open_3d_sourcefiles
from io_tools.open_vectorfiles import *
from kernel_wrappers.Kernel3D import Kernel3D, Kernel3DConfig


def run_advector_3D(
    sourcefile_path: str,
    configfile_path: str,
    output_directory: str,
    u_water_path: Union[List[str], str],
    v_water_path: Union[List[str], str],
    w_water_path: Union[List[str], str],
    seawater_density_path: Union[List[str], str],
    advection_start_date: datetime.datetime,
    timestep: datetime.timedelta,
    num_timesteps: int,
    advection_scheme: str = "taylor2",
    save_period: int = 1,
    opencl_device: Tuple[int, ...] = None,
    memory_utilization: float = 0.4,
    u_wind_path: Optional[Union[List[str], str]] = None,
    v_wind_path: Optional[Union[List[str], str]] = None,
    windage_multiplier: float = 1,
    wind_mixing_enabled: bool = True,
    show_progress_bar: bool = True,
    water_preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
    wind_preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
    seawater_density_preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
    sourcefile_preprocessor: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
    overwrite_existing_files: bool = False,
) -> List[str]:
    """
    :param sourcefile_path: path to the particle sourcefile netcdf file.
        Can be a wildcard path as long as the individual sourcefiles can be properly concatenated along particle axis.
        See forcing_data_specifications.md for data requirements.
    :param configfile_path: path to the configfile netcdf file.
        See configfile_specifications.md for details
    :param output_directory: directory which will be populated with the outfiles.
        Existing files in this directory may be overwritten.
        See forcing_data_specifications.md for outputfile format details.
    :param u_water_path: wildcard path to the zonal current files.
        See forcing_data_specifications.md for data requirements.
    :param v_water_path: wildcard path to the meridional current files; see 'u_water_path'.
    :param w_water_path: wildcard path to the vertical current files; see 'u_water_path'.
    :param seawater_density_path: wildcard path to the seawater seawater_density files.
        See forcing_data_specifications.md for data requirements.
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
    :param windage_multiplier: multiplies the default windage, which is based on emerged area.
    :param wind_mixing_enabled: enable/disable near-surface turbulent wind mixing.
    :param show_progress_bar: whether to show progress bars for dask operations
    :param water_preprocessor: function to manipulate the water data just after loading.
        After preprocessor is applied, data must be compliant with forcing_data_specifications.md
    :param wind_preprocessor: see water_preprocessor
    :param seawater_density_preprocessor: see water_preprocessor
    :param sourcefile_preprocessor: see water_preprocessor, compliance info in sourcefile_specifications.md
    :param overwrite_existing_files: flag to skip warning prompts and clobber existing files,
        useful for running model with no possibility of user input
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
    p0 = open_3d_sourcefiles(
        sourcefile_path=sourcefile_path,
        preprocessor=sourcefile_preprocessor,
    )

    print("Opening Configfile...")
    eddy_diffusivity, max_wave_height, wave_mixing_depth_factor = unpack_configfile(
        configfile_path=configfile_path
    )

    forcing_data = {}
    print("Initializing Ocean Current...")
    forcing_data[Forcing.current] = open_3d_currents(
        u_path=u_water_path,
        v_path=v_water_path,
        w_path=w_water_path,
        preprocessor=water_preprocessor,
    )

    print("Initializing Seawater Density...")
    forcing_data[Forcing.seawater_density] = open_seawater_density(
        path=seawater_density_path,
        preprocessor=seawater_density_preprocessor,
    )

    if u_wind_path is not None and v_wind_path is not None:
        print("Initializing Wind...")
        forcing_data[Forcing.wind] = open_wind(
            u_path=u_wind_path, v_path=v_wind_path, preprocessor=wind_preprocessor
        )

    output_writer = OutputWriter3D(
        out_dir=Path(output_directory),
        basename="ADVECTOR_3D_output",
        configfile=xr.open_dataset(configfile_path),
        sourcefile=p0,
        forcing_data=forcing_data,
        api_entry="src/run_advector_3D.py::run_advector_3D",
        api_arguments=arguments,
        overwrite_existing_files=overwrite_existing_files,
    )

    print("---COMMENCING ADVECTION---")
    out_paths = execute_chunked_kernel_computation(
        forcing_data=forcing_data,
        kernel_cls=Kernel3D,
        kernel_config=Kernel3DConfig(
            advection_scheme=scheme_enum,
            eddy_diffusivity=eddy_diffusivity,
            max_wave_height=max_wave_height,
            wave_mixing_depth_factor=wave_mixing_depth_factor,
            windage_multiplier=windage_multiplier,
            wind_mixing_enabled=wind_mixing_enabled,
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
