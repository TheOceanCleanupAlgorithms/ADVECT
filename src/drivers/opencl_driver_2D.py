import datetime

import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd

from typing import Tuple
from dask.diagnostics import ProgressBar
from drivers.advection_chunking import chunk_advection_params
from kernel_wrappers.Kernel2D import Kernel2D, AdvectionScheme


def openCL_advect(field: xr.Dataset,
                  p0: pd.DataFrame,
                  start_time: datetime.datetime,
                  dt: datetime.timedelta,
                  num_timesteps: int,
                  save_every: int,
                  advection_scheme: AdvectionScheme,
                  eddy_diffusivity: float,
                  platform_and_device: Tuple[int, int] = None,
                  verbose=False) -> Tuple[xr.Dataset, float, float]:
    """
    advect particles on device using OpenCL.  Dynamically chunks computation to fit device memory.
    :param field: xarray Dataset storing vector field/axes.
                    Dimensions: {'time', 'lon', 'lat'}
                    Variables: {'U', 'V'}
    :param p0: initial positions of particles, pandas dataframe with columns ['lon', 'lat', 'release_date']
    :param start_time: advection start time
    :param dt: timestep duration
    :param num_timesteps: number of timesteps
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param advection_scheme: scheme to use, listed in the AdvectionScheme enum
    :param eddy_diffusivity: constant, scales random walk, model dependent value
    :param platform_and_device: indices of platform/device to execute program.  None initiates interactive mode.
    :param verbose: determines whether to print buffer sizes and timing results
    :return: (P, buffer_seconds, kernel_seconds): (numpy array with advection paths, shape (num_particles, num_timesteps, 2),
                                                   time it took to transfer memory to/from device,
                                                   time it took to execute kernel on device)
    """
    num_particles = len(p0)
    advect_time = pd.date_range(start=start_time, freq=dt, periods=num_timesteps)

    # choose the device/platform we're running on
    if platform_and_device is None:
        context = cl.create_some_context(interactive=True)
    else:
        context = cl.create_some_context(answers=list(platform_and_device))

    # get the minimum RAM available on the specified compute devices.
    available_RAM = min(device.global_mem_size for device in context.devices) * .95  # leave 5% for safety
    advect_time_chunks, out_time_chunks, field_chunks = \
        chunk_advection_params(available_RAM, field, num_particles, advect_time, save_every)

    buf_time, kernel_time = 0, 0
    P_chunks = []
    p0_chunk = p0
    for advect_time_chunk, out_time_chunk, field_chunk in zip(advect_time_chunks, out_time_chunks, field_chunks):
        print(f'Chunk {len(P_chunks)+1:3}/{len(field_chunks)}: '
              f'{field_chunk.time.values[0]} to {field_chunk.time.values[-1]}...')
        print(f'  Loading currents...')
        with ProgressBar():
            field_chunk.load()

        num_timesteps_chunk = len(advect_time_chunk) - 1  # because initial position is given!
        out_timesteps_chunk = len(out_time_chunk) - 1     #
        # create the kernel wrapper object, pass it arguments
        kernel = create_kernel(advection_scheme=advection_scheme, eddy_diffusivity=eddy_diffusivity,
                               context=context, field=field_chunk, p0=p0_chunk, num_particles=num_particles,
                               dt=dt, start_time=advect_time_chunk[0], num_timesteps=num_timesteps_chunk, save_every=save_every,
                               out_timesteps=out_timesteps_chunk)
        kernel.execute()

        buf_time += kernel.buf_time
        kernel_time += kernel.kernel_time
        if verbose:
            kernel.print_memory_footprint()
            kernel.print_execution_time()

        P_chunk = create_dataset_from_kernel(kernel, out_time_chunk)
        if len(P_chunks) > 0:  # except for first chunk, leave out p0
            P_chunk = P_chunk.isel(time=slice(1, None))
        P_chunks.append(P_chunk)
        p0_chunk = P_chunk.isel(time=-1).to_dataframe()

    P = xr.concat(P_chunks, dim='time')

    return P, buf_time, kernel_time


def create_kernel(advection_scheme: AdvectionScheme, eddy_diffusivity: float,
                  context: cl.Context, field: xr.Dataset, p0: pd.DataFrame,
                  num_particles: int, dt: datetime.timedelta, start_time: pd.Timestamp,
                  num_timesteps: int, save_every: int, out_timesteps: int) -> Kernel2D:
    """create and return the wrapper for the opencl kernel"""
    field = field.transpose('time', 'lon', 'lat')

    return Kernel2D(
            advection_scheme=advection_scheme,
            eddy_diffusivity=eddy_diffusivity,
            context=context,
            field_x=field.lon.values.astype(np.float64),
            field_y=field.lat.values.astype(np.float64),
            field_t=field.time.values.astype('datetime64[s]').astype(np.float64),  # float64 representation of unix timestamp
            field_U=field.U.values.astype(np.float32).flatten(),
            field_V=field.V.values.astype(np.float32).flatten(),
            x0=p0.lon.values.astype(np.float32),
            y0=p0.lat.values.astype(np.float32),
            release_date=p0['release_date'].values.astype('datetime64[s]').astype(np.float64),
            start_time=start_time.timestamp(),
            dt=dt.total_seconds(),
            ntimesteps=num_timesteps,
            save_every=save_every,
            X_out=np.full((num_particles*out_timesteps), np.nan).astype(np.float32),  # output will have this value
            Y_out=np.full((num_particles*out_timesteps), np.nan).astype(np.float32),  # unless overwritten (e.g. pre-release)
    )


def create_dataset_from_kernel(kernel: Kernel2D, advect_time: pd.DatetimeIndex) -> xr.Dataset:
    """assumes kernel has been run"""
    num_particles = len(kernel.x0)
    lon = kernel.X_out.reshape([num_particles, -1])
    lat = kernel.Y_out.reshape([num_particles, -1])

    P = xr.Dataset(data_vars={'lon': (['p_id', 'time'], lon),
                              'lat': (['p_id', 'time'], lat)},
                   coords={'p_id': np.arange(num_particles),
                           'time': advect_time[1:]}  # initial positions are not returned
                   )
    return P
