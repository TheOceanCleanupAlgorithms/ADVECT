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


def openCL_advect(current: xr.Dataset,
                  wind: xr.Dataset,
                  out_dir: Path,
                  p0: xr.Dataset,
                  start_time: datetime.datetime,
                  dt: datetime.timedelta,
                  num_timesteps: int,
                  save_every: int,
                  advection_scheme: AdvectionScheme,
                  eddy_diffusivity: float,
                  windage_multiplier: Optional[float],
                  memory_utilization: float,
                  platform_and_device: Tuple[int] = None,
                  verbose=False) -> List[Path]:
    """
    advect particles on device using OpenCL.  Dynamically chunks computation to fit device memory.
    :param current: xarray Dataset storing current vector field/axes.
    :param wind: xarray Dataset storing wind vector field/axes.  If None, no windage applied.
    :param out_dir: directory in which to save the outputfiles
    :param p0: xarray Dataset storing particle initial state from sourcefile
    :param start_time: advection start time
    :param dt: timestep duration
    :param num_timesteps: number of timesteps
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param advection_scheme: scheme to use, listed in the AdvectionScheme enum
    :param eddy_diffusivity: constant, scales random walk, model dependent value
    :param windage_multiplier: multiplies the default windage, which is based on emerged area
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
    advect_time_chunks, out_time_chunks, current_chunks, wind_chunks = \
        chunk_advection_params(device_bytes=available_RAM,
                               current=current,
                               wind=wind,
                               num_particles=num_particles,
                               advect_time=advect_time,
                               save_every=save_every)

    buf_time, kernel_time = 0, 0
    writer = OutputWriter(out_dir=out_dir)
    create_logger(out_dir / "warnings.log")
    p0_chunk = p0.assign({'exit_code': ('p_id', np.zeros(len(p0.p_id)))})
    for i, (advect_time_chunk, out_time_chunk, current_chunk, wind_chunk) \
            in enumerate(zip(advect_time_chunks, out_time_chunks, current_chunks, wind_chunks)):
        print(f'Chunk {i+1:3}/{len(current_chunks)}: '
              f'{current_chunk.time.values[0]} to {current_chunk.time.values[-1]}...')

        num_timesteps_chunk = len(advect_time_chunk) - 1  # because initial position is given!
        out_timesteps_chunk = len(out_time_chunk) - 1     #

        # create the kernel wrapper object, pass it arguments
        with ProgressBar():
            print(f'  Loading currents and wind...')  # these get implicitly loaded when .values is called on current_chunk variables
            kernel = create_kernel(advection_scheme=advection_scheme, eddy_diffusivity=eddy_diffusivity, windage_multiplier=windage_multiplier,
                                   context=context, current=current_chunk, wind=wind_chunk, p0=p0_chunk, num_particles=num_particles,
                                   dt=dt, start_time=advect_time_chunk[0], num_timesteps=num_timesteps_chunk, save_every=save_every,
                                   out_timesteps=out_timesteps_chunk)
        kernel.execute()

        buf_time += kernel.buf_time
        kernel_time += kernel.kernel_time
        if verbose:
            kernel.print_memory_footprint()
            kernel.print_execution_time()

        P_chunk = create_dataset_from_kernel(kernel=kernel,
                                             chunk_init_state=p0_chunk,
                                             advect_time=out_time_chunk)
        handle_errors(chunk=P_chunk, chunk_num=i + 1)

        del kernel  # important for releasing memory for the next iteration
        gc.collect()

        writer.write_output_chunk(P_chunk)

        p0_chunk = P_chunk.isel(time=-1)  # last timestep is initial state for next chunk
        # problem is, this ^ has nans for location of all the unreleased particles.  Restore that information here
        unreleased = p0_chunk.release_date > advect_time_chunk[-1]
        for var in ['lat', 'lon', 'depth']:
            p0_chunk[var].loc[unreleased] = p0[var].loc[unreleased]

    return writer.paths


def create_kernel(advection_scheme: AdvectionScheme, eddy_diffusivity: float, windage_multiplier: float,
                  context: cl.Context, current: xr.Dataset, wind: xr.Dataset, p0: xr.Dataset,
                  num_particles: int, dt: datetime.timedelta, start_time: pd.Timestamp,
                  num_timesteps: int, save_every: int, out_timesteps: int) -> Kernel3D:
    """create and return the wrapper for the opencl kernel"""
    current = current.transpose('time', 'depth', 'lon', 'lat')
    wind = wind.transpose('time', 'lon', 'lat')
    return Kernel3D(
            advection_scheme=advection_scheme,
            eddy_diffusivity=eddy_diffusivity,
            windage_multiplier=windage_multiplier,
            context=context,
            current_x=current.lon.values.astype(np.float64),
            current_y=current.lat.values.astype(np.float64),
            current_z=current.depth.values.astype(np.float64),
            current_t=current.time.values.astype('datetime64[s]').astype(np.float64),  # float64 representation of unix timestamp
            current_U=current.U.values.astype(np.float32, copy=False).ravel(),  # astype will still copy if field.U is not already float32
            current_V=current.V.values.astype(np.float32, copy=False).ravel(),
            current_W=current.W.values.astype(np.float32, copy=False).ravel(),
            wind_x=wind.lon.values.astype(np.float64),
            wind_y=wind.lat.values.astype(np.float64),
            wind_t=wind.time.values.astype('datetime64[s]').astype(np.float64),  # float64 representation of unix timestamp
            wind_U=wind.U.values.astype(np.float32, copy=False).ravel(),  # astype will still copy if field.U is not already float32
            wind_V=wind.V.values.astype(np.float32, copy=False).ravel(),
            x0=p0.lon.values.astype(np.float32),
            y0=p0.lat.values.astype(np.float32),
            z0=p0.depth.values.astype(np.float32),
            release_date=p0.release_date.values.astype('datetime64[s]').astype(np.float64),
            radius=p0.radius.values.astype(np.float64),
            density=p0.density.values.astype(np.float64),
            start_time=start_time.timestamp(),
            dt=dt.total_seconds(),
            ntimesteps=num_timesteps,
            save_every=save_every,
            X_out=np.full((num_particles*out_timesteps), np.nan, dtype=np.float32),  # output will have this value
            Y_out=np.full((num_particles*out_timesteps), np.nan, dtype=np.float32),  # unless overwritten (e.g. pre-release)
            Z_out=np.full((num_particles*out_timesteps), np.nan, dtype=np.float32),
            exit_code=p0.exit_code.values.astype(np.byte),
    )


def create_dataset_from_kernel(
    kernel: Kernel3D, chunk_init_state: xr.Dataset, advect_time: pd.DatetimeIndex
) -> xr.Dataset:
    """assumes kernel has been run"""
    P = chunk_init_state.assign_coords({"time": advect_time[1:]})  # add a time dimension

    P = P.assign(  # overwrite with new data
        {
            "lon": (["p_id", "time"], kernel.X_out.reshape([len(P.p_id), -1])),
            "lat": (["p_id", "time"], kernel.Y_out.reshape([len(P.p_id), -1])),
            "depth": (["p_id", "time"], kernel.Z_out.reshape([len(P.p_id), -1])),
            "exit_code": (["p_id"], kernel.exit_code),
        }
    )
    return P


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
