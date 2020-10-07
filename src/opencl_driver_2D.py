import pyopencl as cl
import numpy as np
import time
import xarray as xr
import pandas as pd
import math

from typing import Tuple
from kernels.eulerian_kernel_2d_check_host_args import check_args


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
    context, queue, program = setup_opencl_objects(platform_and_device, kernel_file='kernels/eulerian_kernel_2d.cl')

    # get the minimum RAM available on the specified compute devices.
    available_RAM = min(device.global_mem_size for device in context.devices) * .95  # leave 5% for safety

    advect_time_chunks, out_time_chunks, field_chunks = \
        chunk_advection(available_RAM, field, p0, advect_time, save_every)

    num_particles = len(p0)
    num_timesteps = len(advect_time) - 1  # because initial position is given!
    t0 = advect_time[0].timestamp()  # unix timestamp
    dt = (advect_time[1] - advect_time[0]).total_seconds()
    out_timesteps = num_timesteps // save_every

    # perform the basic steps of the advection calculation, leaving details up to subfunctions
    host_bufs, device_bufs, write_time = create_buffers(context, field, p0, out_timesteps, t0, verbose)
    kernel_timer = execute_kernel(program, queue, host_bufs, device_bufs, num_particles, dt, num_timesteps, save_every)
    read_time, kernel_time = read_buffers_back_from_kernel(queue, host_bufs, device_bufs, kernel_timer)

    # store results in Dataset
    P = create_dataset_from_advection_output(p0, host_bufs, num_particles, out_timesteps, advect_time, save_every)

    buf_time = write_time + read_time
    if verbose:
        print(f'memory operations took {buf_time: .3f} seconds')
        print(f'kernel execution took  {kernel_time: .3f} seconds')

    return P, buf_time, kernel_time


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


def setup_opencl_objects(platform_and_device, kernel_file):
    """setup the objects which control the computation"""
    if platform_and_device is None:
        context = cl.create_some_context(interactive=True)
    else:
        context = cl.create_some_context(answers=list(platform_and_device))

    # Create a command queue
    queue = cl.CommandQueue(context)

    # Create and build the kernel from the source code
    program = cl.Program(context, open(kernel_file).read()).build()

    return context, queue, program


def create_buffers(context, field, p0, out_timesteps, t0, verbose):
    """here we create host and device buffers, and return them as dictionaries"""
    host_bufs = create_host_buffers(field, p0, out_timesteps, t0)
    if verbose:  # print size of buffers
        for buf_name, buf_value in host_bufs.items():
            print(f'{buf_name}: {buf_value.nbytes / 1e6} MB')
    device_bufs, write_time = create_device_buffers(context, host_bufs)
    return host_bufs, device_bufs, write_time


def create_host_buffers(field, p0, out_timesteps, t0):
    # initialize host vectors.  These are the host-side arguments for the kernel.
    field = field.transpose('time', 'lon', 'lat')  # make sure the underlying numpy arrays are in the correct shape
    h_field_x = field.lon.values.astype(np.float64)
    h_field_y = field.lat.values.astype(np.float64)
    h_field_t = field.time.values.astype('datetime64[s]').astype(np.float64)  # float64 representation of unix timestamp
    h_field_U = field.U.values.astype(np.float32).flatten()
    h_field_V = field.V.values.astype(np.float32).flatten()
    h_x0 = p0.lon.values.astype(np.float32)
    h_y0 = p0.lat.values.astype(np.float32)
    h_t0 = (t0 * np.ones(len(p0))).astype(np.float32)
    h_X_out = np.zeros(len(p0) * out_timesteps).astype(np.float32)
    h_Y_out = np.zeros(len(p0) * out_timesteps).astype(np.float32)
    host_bufs = {'h_field_x': h_field_x, 'h_field_y': h_field_y, 'h_field_t': h_field_t,
                 'h_field_U': h_field_U, 'h_field_V': h_field_V,
                 'h_x0': h_x0, 'h_y0': h_y0, 'h_t0': h_t0,
                 'h_X_out': h_X_out, 'h_Y_out': h_Y_out}
    return host_bufs


def create_device_buffers(context, host_bufs):
    # Create the input arrays in device memory and copy data from host
    tic = time.time()
    d_field_x, d_field_y, d_field_t, d_field_U, d_field_V, d_x0, d_y0, d_t0 = \
        (cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)
         for hostbuf in (host_bufs['h_field_x'], host_bufs['h_field_y'], host_bufs['h_field_t'],
                         host_bufs['h_field_U'], host_bufs['h_field_V'],
                         host_bufs['h_x0'], host_bufs['h_y0'], host_bufs['h_t0']))
    # Create the output arrays in device memory
    d_X_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_bufs['h_X_out'].nbytes)
    d_Y_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_bufs['h_Y_out'].nbytes)
    device_bufs = {'d_field_x': d_field_x, 'd_field_y': d_field_y, 'd_field_t': d_field_t,
                   'd_field_U': d_field_U, 'd_field_V': d_field_V,
                   'd_x0': d_x0, 'd_y0': d_y0, 'd_t0': d_t0,
                   'd_X_out': d_X_out, 'd_Y_out': d_Y_out}
    write_time = time.time() - tic
    return device_bufs, write_time


def execute_kernel(program, queue, host_bufs, device_bufs, num_particles, dt, num_timesteps, save_every):
    """this function is responsible for passing args to and executing the kernel."""
    # CHECK KERNEL ARGUMENTS
    check_args(host_bufs)

    # Execute the kernel, allowing OpenCL runtime to select the work group items for the device

    advect = program.advect
    advect.set_scalar_arg_dtypes([None, np.uint32, None, np.uint32, None, np.uint32,
                                  None, None,
                                  None, None, None,
                                  np.float32, np.uint32, np.uint32,
                                  None, None])
    kernel_timer = time.time()
    advect(queue, (num_particles,), None,
           device_bufs['d_field_x'], np.uint32(len(host_bufs['h_field_x'])),
           device_bufs['d_field_y'], np.uint32(len(host_bufs['h_field_y'])),
           device_bufs['d_field_t'], np.uint32(len(host_bufs['h_field_t'])),
           device_bufs['d_field_U'], device_bufs['d_field_V'],
           device_bufs['d_x0'], device_bufs['d_y0'], device_bufs['d_t0'],
           np.float32(dt), np.uint32(num_timesteps), np.uint32(save_every),
           device_bufs['d_X_out'], device_bufs['d_Y_out'])
    return kernel_timer


def read_buffers_back_from_kernel(queue, host_bufs, device_bufs, kernel_timer):
    """blocks until command queue has finished. Reads back the output buffers"""
    # wait for the computation to complete
    queue.finish()
    kernel_time = time.time() - kernel_timer

    # Read back the results from the compute device
    tic = time.time()
    cl.enqueue_copy(queue, host_bufs['h_X_out'], device_bufs['d_X_out'])
    cl.enqueue_copy(queue, host_bufs['h_Y_out'], device_bufs['d_Y_out'])
    read_time = time.time() - tic

    return read_time, kernel_time


def create_dataset_from_advection_output(p0, host_bufs, num_particles, out_timesteps, advect_time, save_every):
    lon = np.concatenate([p0.lon.values[:, np.newaxis],
                          host_bufs['h_X_out'].reshape([num_particles, out_timesteps])], axis=1)
    lat = np.concatenate([p0.lat.values[:, np.newaxis],
                          host_bufs['h_Y_out'].reshape([num_particles, out_timesteps])], axis=1)
    P = xr.Dataset(data_vars={'lon': (['p_id', 'time'], lon),
                              'lat': (['p_id', 'time'], lat)},
                   coords={'p_id': np.arange(num_particles),
                           'time': advect_time[::save_every]})
    return P
