import datetime
import gc
import logging
from pathlib import Path

import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd

from typing import Tuple, Optional, List
from dask.diagnostics import ProgressBar
from drivers.advection_chunking import chunk_advection_params
from io_tools.OutputWriter import OutputWriter
from kernel_wrappers.Kernel3D import Kernel3D, AdvectionScheme
from kernel_wrappers.kernel_constants import EXIT_CODES


def openCL_advect(
    current: xr.Dataset,
    wind: xr.Dataset,
    seawater_density: xr.Dataset,
    output_writer: OutputWriter,
    p0: xr.Dataset,
    start_time: datetime.datetime,
    dt: datetime.timedelta,
    num_timesteps: int,
    save_every: int,
    advection_scheme: AdvectionScheme,
    eddy_diffusivity: xr.Dataset,
    max_wave_height: float,
    wave_mixing_depth_factor: float,
    windage_multiplier: Optional[float],
    wind_mixing_enabled: bool,
    memory_utilization: float,
    platform_and_device: Tuple[int] = None,
    verbose=False,
) -> List[Path]:
    """
    advect particles on device using OpenCL.  Dynamically chunks computation to fit device memory.
    :param current: xarray Dataset storing current vector field/axes.
    :param wind: xarray Dataset storing wind vector field/axes.  If None, no windage applied.
    :param seawater_density: xarray Dataset storing seawater density field.
    :param output_writer: object which is responsible for persisting the model output to disk
    :param p0: xarray Dataset storing particle initial state from sourcefile
    :param start_time: advection start time
    :param dt: timestep duration
    :param num_timesteps: number of timesteps
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param advection_scheme: scheme to use, listed in the AdvectionScheme enum
    :param eddy_diffusivity: xarray Dataset storing vertical profiles of eddy diffusivities
    :param density_profile: xarray Dataset storing vertical profile of seawater density
    :param max_wave_height: caps parameterization in kernel; see config_specifications.md
    :param wave_mixing_depth_factor: scales depth of mixing in kernel; see config_specifications.md
    :param windage_multiplier: multiplies the default windage, which is based on emerged area
    :param wind_mixing_enabled: toggle the wind mixing functionality
    :param memory_utilization: fraction of the opencl device memory available for buffers
    :param platform_and_device: indices of platform/device to execute program.  None initiates interactive mode.
    :param verbose: determines whether to print buffer sizes and timing results
    :return: list of outputfile paths
    """
    num_particles = len(p0.p_id)
    advect_time = pd.date_range(start=start_time, freq=dt, periods=num_timesteps)
    current = current.sel(time=slice(advect_time[0], advect_time[-1]))  # trim vector fields to necessary time range
    wind = wind.sel(time=slice(advect_time[0], advect_time[-1]))

    # choose the device/platform we're running on
    if platform_and_device is None:
        context = cl.create_some_context(interactive=True)
    else:
        context = cl.create_some_context(answers=list(platform_and_device))

    # get the minimum RAM available on the specified compute devices.
    available_RAM = min(device.global_mem_size for device in context.devices) * memory_utilization
    advect_time_chunks, current_chunks, wind_chunks, seawater_density_chunks = \
        chunk_advection_params(device_bytes=available_RAM,
                               current=current,
                               wind=wind,
                               seawater_density=seawater_density,
                               num_particles=num_particles,
                               advect_time=advect_time,
                               save_every=save_every)

    create_logger(output_writer.folder_path / "warnings.log")
    p0_chunk = p0.assign({'exit_code': ('p_id', np.zeros(len(p0.p_id)))})
    for i, (advect_time_chunk, current_chunk, wind_chunk, seawater_density_chunk) \
            in enumerate(zip(advect_time_chunks, current_chunks, wind_chunks, seawater_density_chunks)):
        print(f'Chunk {i+1:3}/{len(current_chunks)}: '
              f'{current_chunk.time.values[0]} to {current_chunk.time.values[-1]}...')

        # create the kernel wrapper object, pass it arguments
        with ProgressBar():
            print(f'  Loading forcing data...')   # these get implicitly loaded when .values is called on current_chunk variables
            kernel = Kernel3D(
                current=current_chunk,
                wind=wind_chunk,
                seawater_density=seawater_density_chunk,
                p0=p0_chunk,
                advection_scheme=advection_scheme,
                eddy_diffusivity=eddy_diffusivity,
                max_wave_height=max_wave_height,
                wave_mixing_depth_factor=wave_mixing_depth_factor,
                windage_multiplier=windage_multiplier,
                wind_mixing_enabled=wind_mixing_enabled,
                advect_time=advect_time_chunk,
                save_every=save_every,
                context=context)
        P_chunk = kernel.execute()
        handle_errors(chunk=P_chunk, chunk_num=i + 1)

        if verbose:
            kernel.print_memory_footprint()
            kernel.print_execution_time()

        del kernel  # important for releasing memory for the next iteration
        gc.collect()

        output_writer.write_output_chunk(P_chunk)

        p0_chunk = P_chunk.isel(time=-1)  # last timestep is initial state for next chunk
        # problem is, this ^ has nans for location of all the unreleased particles.  Restore that information here
        unreleased = p0_chunk.release_date > advect_time_chunk[-1]
        for var in ['lat', 'lon', 'depth']:
            p0_chunk[var].loc[unreleased] = p0[var].loc[unreleased]

    return output_writer.paths


def create_logger(log_path: Path):
    """this sets up logging such that logs with level WARNING go to log_path,
        logs with level ERROR or greater go to log_path and stdout."""
    logging.basicConfig(filename=str(log_path), filemode='w', level=logging.WARNING,
                        format="%(asctime)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(console)


def handle_errors(chunk: xr.Dataset, chunk_num: int):
    if not np.all(chunk.exit_code == 0):
        bad_codes = np.unique(chunk.exit_code[chunk.exit_code != 0])
        logging.error(f"Error: {np.count_nonzero(chunk.exit_code)} particle(s) did not exit successfully: "
                      f"exit code(s) {[f'{code} ({EXIT_CODES[code]})' for code in bad_codes]}")
        for i, code in enumerate(chunk.exit_code[chunk.exit_code != 0].values):
            logging.warning(f"Chunk {chunk_num: 3}: Particle ID {chunk.p_id.values[i]} exited with error code {code}.")
    if np.any(chunk.exit_code < 0):
        raise ValueError(f"Fatal error encountered, error code(s) "
                         f"{np.unique(chunk.exit_code[chunk.exit_code < 0])}; aborting")
