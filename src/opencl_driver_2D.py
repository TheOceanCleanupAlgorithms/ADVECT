import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd
import math

from typing import Tuple
from kernels.EulerianKernel2D import EulerianKernel2D


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
    num_timesteps = len(advect_time) - 1  # because initial position is given!
    t0 = advect_time[0]
    dt = (advect_time[1] - advect_time[0]).total_seconds()
    out_timesteps = num_timesteps // save_every

    # choose the device/platform we're running on
    if platform_and_device is None:
        context = cl.create_some_context(interactive=True)
    else:
        context = cl.create_some_context(answers=list(platform_and_device))

    # get the minimum RAM available on the specified compute devices.
    available_RAM = min(device.global_mem_size for device in context.devices) * .95  # leave 5% for safety

    advect_time_chunks, out_time_chunks, field_chunks = \
        chunk_advection(available_RAM, field, p0, advect_time, save_every)

    # create the kernel wrapper object, pass it arguments
    kernel = create_kernel(context, field=field, p0=p0, num_particles=num_particles,
                           dt=dt, t0=t0, num_timesteps=num_timesteps, save_every=save_every,
                           out_timesteps=out_timesteps)
    kernel.execute()

    if verbose:
        kernel.print_memory_footprint()
        kernel.print_execution_time()

    P = create_dataset_from_kernel(kernel, advect_time[::save_every])

    return P, kernel.buf_time, kernel.kernel_time


def create_kernel(context: cl.Context, field: xr.Dataset, p0: pd.DataFrame,
                  num_particles: int, dt: float, t0: pd.Timestamp,
                  num_timesteps: int, save_every: int, out_timesteps: int) -> EulerianKernel2D:
    """create and return the wrapper for the opencl kernel"""
    field = field.transpose('time', 'lon', 'lat')

    return EulerianKernel2D(
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


def chunk_advection(device_bytes, field, p0, advect_time, save_every):
    out_time = advect_time[::save_every]
    # each element of out_time marks a time at which the driver will return particle position

    # estimate total size of memory we need to eventually run through the device
    field_bytes, output_bytes, p0_bytes = estimate_memory_bytes(field, len(p0), len(out_time)-1)
    available_bytes_for_field = device_bytes - (output_bytes + p0_bytes)
    num_chunks = math.ceil(field_bytes / available_bytes_for_field)  # minimum chunking to potentially fit RAM

    # it's hard to pre-compute the exact number of chunks necessary that will fit into RAM.  we start with a good guess,
    # but if any of the chunks we create don't fit into RAM, just increment the number of chunks and try again.
    while True:
        # now we split up the advection OUTPUT into chunks.  All else will be based on this splitting.
        assert len(out_time) >= num_chunks, 'Cannot split computation, output frequency is too low!'
        # the above situation arises when the span of time between particle save points corresponds to a chunk of field
        # which is too large to fit onto the compute device.

        chunk_len = math.ceil(len(out_time) / num_chunks)
        out_time_chunks = [out_time[i*chunk_len: (i+1)*chunk_len + 1] for i in range(num_chunks)]
        advect_time_chunks = [advect_time[(out_time_chunk[0] <= advect_time) & (advect_time <= out_time_chunk[-1])]
                              for out_time_chunk in out_time_chunks]
        # subsequent time chunks have overlapping endpoints.  This is because the final reported value
        # from a computation will be fed to the next computation as the start point, at the same time.

        field_chunks = [field.sel(time=slice(out_time_chunk[0], out_time_chunk[-1]))
                        for out_time_chunk in out_time_chunks]

        if all(device_bytes-sum(estimate_memory_bytes(field_chunk, len(p0), len(out_time_chunk)-1)) > 0
               for field_chunk, out_time_chunk in zip(field_chunks, out_time_chunks)):
            break
        num_chunks += 1

    return advect_time_chunks, out_time_chunks, field_chunks


def estimate_memory_bytes(field, num_particles, out_timesteps):
    """This estimates total memory needed for the buffers.
    There's a bit more needed for the scalar arguments, but this is tiny"""
    field_bytes = (2 * 4 * np.prod(field.U.shape) +  # two 32-bit fields
                   8 * (len(field.lon) + len(field.lat) + len(field.time)))  # the 3 64-bit coordinate arrays
    output_bytes = 2 * 4 * num_particles * out_timesteps   # two 32-bit variables for each particle for each timestep
    p0_bytes = 2 * 4 * num_particles  # two 32-bit variables for each particle
    return field_bytes, output_bytes, p0_bytes
