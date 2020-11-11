import datetime
import gc
from pathlib import Path

import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd

from typing import Tuple, Optional
from dask.diagnostics import ProgressBar
from drivers.advection_chunking import chunk_advection_params
from io_tools.write_to_outputfile import OutputWriter
from kernel_wrappers.Kernel3D import Kernel3D, AdvectionScheme3D


def openCL_advect(current: xr.Dataset,
                  wind: xr.Dataset,
                  out_path: Path,
                  p0: pd.DataFrame,
                  start_time: datetime.datetime,
                  dt: datetime.timedelta,
                  num_timesteps: int,
                  save_every: int,
                  advection_scheme: AdvectionScheme3D,
                  eddy_diffusivity: float,
                  windage_coeff: Optional[float],
                  memory_utilization: float,
                  platform_and_device: Tuple[int] = None,
                  verbose=False) -> Tuple[float, float]:
    """
    advect particles on device using OpenCL.  Dynamically chunks computation to fit device memory.
    :param current: xarray Dataset storing current vector field/axes.
                     Dimensions: {'time', 'lon', 'lat'}
                     Variables: {'U', 'V'}
    :param wind: xarray Dataset storing wind vector field/axes.  If None, no windage applied.
                 Dimensions: {'time', 'lon', 'lat'}
                 Variables: {'U', 'V'}
    :param out_path: path at which to save the outputfile
    :param p0: initial positions of particles, pandas dataframe with columns ['lon', 'lat', 'release_date']
    :param start_time: advection start time
    :param dt: timestep duration
    :param num_timesteps: number of timesteps
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param advection_scheme: scheme to use, listed in the AdvectionScheme enum
    :param eddy_diffusivity: constant, scales random walk, model dependent value
    :param windage_coeff: constant in [0, 1], fraction of windspeed applied to particle
    :param memory_utilization: fraction of the opencl device memory available for buffers
    :param platform_and_device: indices of platform/device to execute program.  None initiates interactive mode.
    :param verbose: determines whether to print buffer sizes and timing results
    :return: (buffer_seconds, kernel_seconds): (time it took to transfer memory to/from device,
                                                time it took to execute kernel on device)
    """
    num_particles = len(p0)
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
    writer = OutputWriter(out_path=out_path)
    p0_chunk = p0.copy()
    for i, (advect_time_chunk, out_time_chunk, current_chunk, wind_chunk) \
            in enumerate(zip(advect_time_chunks, out_time_chunks, current_chunks, wind_chunks)):
        print(f'Chunk {i+1:3}/{len(current_chunks)}: '
              f'{current_chunk.time.values[0]} to {current_chunk.time.values[-1]}...')

        num_timesteps_chunk = len(advect_time_chunk) - 1  # because initial position is given!
        out_timesteps_chunk = len(out_time_chunk) - 1     #

        # create the kernel wrapper object, pass it arguments
        with ProgressBar():
            print(f'  Loading currents and wind...')  # these get implicitly loaded when .values is called on current_chunk variables
            kernel = create_kernel(advection_scheme=advection_scheme, eddy_diffusivity=eddy_diffusivity, windage_coeff=windage_coeff,
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
                                             release_date=p0_chunk.release_date,
                                             advect_time=out_time_chunk)
        
        del kernel  # important for releasing memory for the next iteration
        gc.collect()

        writer.write_output_chunk(P_chunk)

        p0_chunk = P_chunk.isel(time=-1).to_dataframe()
        # problem is, this ^ has nans for location of all the unreleased particles.  Restore that information here
        p0_chunk.loc[p0_chunk.release_date > advect_time_chunk[-1], ['lat', 'lon']] = p0[['lat', 'lon']]

    return buf_time, kernel_time


def create_kernel(advection_scheme: AdvectionScheme3D, eddy_diffusivity: float, windage_coeff: float,
                  context: cl.Context, current: xr.Dataset, wind: xr.Dataset, p0: pd.DataFrame,
                  num_particles: int, dt: datetime.timedelta, start_time: pd.Timestamp,
                  num_timesteps: int, save_every: int, out_timesteps: int) -> Kernel3D:
    """create and return the wrapper for the opencl kernel"""
    current = current.transpose('time', 'depth', 'lon', 'lat')
    wind = wind.transpose('time', 'lon', 'lat')
    current = current.isel(depth=0)
    return Kernel3D(
            advection_scheme=advection_scheme,
            eddy_diffusivity=eddy_diffusivity,
            windage_coeff=windage_coeff,
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
            release_date=p0['release_date'].values.astype('datetime64[s]').astype(np.float64),
            start_time=start_time.timestamp(),
            dt=dt.total_seconds(),
            ntimesteps=num_timesteps,
            save_every=save_every,
            X_out=np.full((num_particles*out_timesteps), np.nan, dtype=np.float32),  # output will have this value
            Y_out=np.full((num_particles*out_timesteps), np.nan, dtype=np.float32),  # unless overwritten (e.g. pre-release)
    )


def create_dataset_from_kernel(kernel: Kernel3D, release_date: np.ndarray, advect_time: pd.DatetimeIndex) -> xr.Dataset:
    """assumes kernel has been run"""
    num_particles = len(kernel.x0)
    lon = kernel.X_out.reshape([num_particles, -1])
    lat = kernel.Y_out.reshape([num_particles, -1])

    P = xr.Dataset(data_vars={'lon': (['p_id', 'time'], lon),
                              'lat': (['p_id', 'time'], lat),
                              'release_date': (['p_id'], release_date)},
                   coords={'p_id': np.arange(num_particles),
                           'time': advect_time[1:]}  # initial positions are not returned
                   )
    return P
