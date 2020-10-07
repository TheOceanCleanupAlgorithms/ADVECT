import pyopencl as cl
import numpy as np
import time
import xarray as xr
import pandas as pd

from typing import Tuple
from kernels.eulerian_kernel_check_host_args_2d import check_args


def openCL_advect(field: xr.Dataset,
                  p0: pd.DataFrame,
                  advect_time: pd.DatetimeIndex,
                  save_every: int,
                  platform_and_device: Tuple[int, int] = None,
                  verbose=False) -> Tuple[xr.Dataset, float, float]:
    """
    advect particles on device using OpenCL.  Assumes device memory is big enough to handle it.
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

    # calculate constants associated with advection
    num_particles = len(p0)
    num_timesteps = len(advect_time) - 1  # because initial position is given!
    t0 = advect_time[0].timestamp()  # unix timestamp
    dt = (advect_time[1] - advect_time[0]).total_seconds()
    out_timesteps = num_timesteps // save_every

    # perform the basic steps of the advection calculation, leaving details up to subfunctions
    context, queue, program = setup_opencl_objects(platform_and_device, kernel_file='kernels/eulerian_kernel_2d.cl')
    host_bufs, device_bufs, write_time = create_buffers(context, field, p0, num_particles, out_timesteps, t0, verbose)
    kernel_timer = execute_kernel(program, queue, host_bufs, device_bufs, num_particles, dt, num_timesteps, save_every)
    read_time, kernel_time = read_buffers_back_from_kernel(queue, host_bufs, device_bufs, kernel_timer)

    # store results in Dataset
    P = create_dataset_from_advection_output(p0, host_bufs, num_particles, out_timesteps, advect_time, save_every)

    buf_time = write_time + read_time
    if verbose:
        print(f'memory operations took {buf_time: .3f} seconds')
        print(f'kernel execution took  {kernel_time: .3f} seconds')

    return P, buf_time, kernel_time


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
    host_bufs = create_host_buffers(field, p0, out_timesteps, t0, verbose)
    device_bufs, write_time = create_device_buffers(context, host_bufs)
    return host_bufs, device_bufs, write_time


def create_host_buffers(field, p0, out_timesteps, t0, verbose):
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
    if verbose:
        # print size of buffers
        for buf_name, buf_value in host_bufs.items():
            print(f'{buf_name}: {buf_value.nbytes / 1e6} MB')
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
    d_X_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_X_out.nbytes)
    d_Y_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_Y_out.nbytes)
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
