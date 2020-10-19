import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd

from typing import Tuple
from dask.diagnostics import ProgressBar
from kernels.EulerianKernel2D import EulerianKernel2D
from drivers.advection_chunking import chunk_advection_params

from kernels.Taylor2Kernel2D import Taylor2Kernel2D


def openCL_advect(field: xr.Dataset,
                  p0: pd.DataFrame,
                  advect_time: pd.DatetimeIndex,
                  save_every: int,
                  platform_and_device: Tuple[int, int] = None,
                  verbose=False) -> Tuple[xr.Dataset, float, float]:
    """
    advect particles on device using OpenCL.  Dynamically chunks computation to fit device memory.
    :param field: xarray Dataset storing vector field/axes.
                    Dimensions: {'time', 'lon', 'lat'}
                    Variables: {'U', 'V'}
    :param p0: initial positions of particles, numpy array shape (num_particles, 2)
    :param advect_time: pandas DatetimeIndex corresponding to the timeseries which the particles will be advected over
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param platform_and_device: indices of platform/device to execute program.  None initiates interactive mode.
    :param verbose: determines whether to print buffer sizes and timing results
    :return: (P, buffer_seconds, kernel_seconds): (numpy array with advection paths, shape (num_particles, num_timesteps, 2),
                                                   time it took to transfer memory to/from device,
                                                   time it took to execute kernel on device)
    """
    num_particles = len(p0)
    dt = (advect_time[1] - advect_time[0]).total_seconds()

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

        num_timesteps = len(advect_time_chunk) - 1  # because initial position is given!
        out_timesteps = len(out_time_chunk) - 1     #
        # create the kernel wrapper object, pass it arguments
        kernel = create_kernel(context, field=field_chunk, p0=p0_chunk, num_particles=num_particles,
                               dt=dt, t0=advect_time_chunk[0], num_timesteps=num_timesteps, save_every=save_every,
                               out_timesteps=out_timesteps)
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


def create_kernel(context: cl.Context, field: xr.Dataset, p0: pd.DataFrame,
                  num_particles: int, dt: float, t0: pd.Timestamp,
                  num_timesteps: int, save_every: int, out_timesteps: int) -> Taylor2Kernel2D:
    """create and return the wrapper for the opencl kernel"""
    field = field.transpose('time', 'lon', 'lat')

    return Taylor2Kernel2D(
            context=context,
            field_x=field.lon.values.astype(np.float64),
            field_y=field.lat.values.astype(np.float64),
            field_t=field.time.values.astype('datetime64[s]').astype(np.float64),  # float64 representation of unix timestamp
            field_U=field.U.values.astype(np.float32).flatten(),
            field_V=field.V.values.astype(np.float32).flatten(),
            x0=p0.lon.values.astype(np.float32),
            y0=p0.lat.values.astype(np.float32),
            t0=(t0.timestamp() * np.ones(num_particles)).astype(np.float32),
            dt=dt,
            ntimesteps=num_timesteps,
            save_every=save_every,
            X_out=np.zeros(num_particles*out_timesteps).astype(np.float32),
            Y_out=np.zeros(num_particles*out_timesteps).astype(np.float32),
    )


def create_dataset_from_kernel(kernel: EulerianKernel2D, advect_time: pd.DatetimeIndex) -> xr.Dataset:
    """assumes kernel has been run, assumes simultaneous particle release"""
    num_particles = len(kernel.x0)
    lon = np.concatenate([kernel.x0[:, np.newaxis],
                          kernel.X_out.reshape([num_particles, -1])], axis=1)
    lat = np.concatenate([kernel.y0[:, np.newaxis],
                          kernel.Y_out.reshape([num_particles, -1])], axis=1)
    assert all(kernel.t0 == kernel.t0[0])  # break if all particles not released simultaneously
    P = xr.Dataset(data_vars={'lon': (['p_id', 'time'], lon),
                              'lat': (['p_id', 'time'], lat)},
                   coords={'p_id': np.arange(num_particles),
                           'time': advect_time}
                   )
    return P
