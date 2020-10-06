import pyopencl as cl
import numpy as np
import time
import xarray as xr
import pandas as pd
from kernels.eulerian_kernel_check_host_args_2d import check_args

from typing import Tuple


def openCL_advect(field: xr.Dataset,
                  p0: pd.DataFrame,
                  advect_time: pd.DatetimeIndex,
                  save_every: int,
                  platform_and_device: Tuple[int, int] = None,
                  verbose=False):
    """
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
    field = field.transpose('time', 'lon', 'lat')  # make sure the underlying numpy arrays are in the correct shape
    field['time'] = (field.time - np.datetime64('1970-01-01')) / np.timedelta64(1,
                                                                                's')  # convert field's time axis to unix timestamp
    num_timesteps = len(advect_time) - 1  # because initial position is given!
    t0 = advect_time[0].timestamp()  # unix timestamp
    dt = (advect_time[1] - advect_time[0]).total_seconds()
    num_particles = len(p0)
    out_timesteps = num_timesteps // save_every

    # -----------OPENCL BOILERPLATE------------
    # Create a compute context
    # Ask the user to select a platform/device on the CLI
    if platform_and_device is None:
        context = cl.create_some_context(interactive=True)
    else:
        context = cl.create_some_context(answers=[str(i) for i in platform_and_device])

    # Create a command queue
    queue = cl.CommandQueue(context)

    # Create the compute program from the source buffer
    # and build it
    program = cl.Program(context, open('kernels/eulerian_kernel_2d.cl').read()).build()

    # initialize host vectors
    h_field_x = field.lon.values.astype(np.float64)
    h_field_y = field.lat.values.astype(np.float64)
    h_field_t = field.time.values.astype('datetime64[s]').astype(np.float64)
    h_field_U = field.U.values.astype(np.float32).flatten()
    h_field_V = field.V.values.astype(np.float32).flatten()
    h_x0 = p0.lon.values.astype(np.float32)
    h_y0 = p0.lat.values.astype(np.float32)
    h_t0 = (t0 * np.ones(num_particles)).astype(np.float32)
    h_X_out = np.zeros(num_particles * out_timesteps).astype(np.float32)
    h_Y_out = np.zeros(num_particles * out_timesteps).astype(np.float32)

    if verbose:
        # print size of buffers
        for buf_name, buf_value in {'h_field_x': h_field_x, 'h_field_y': h_field_y, 'h_field_t': h_field_t,
                                    'h_field_U': h_field_U, 'h_field_V': h_field_V,
                                    'h_x0': h_x0, 'h_y0': h_y0, 'h_t0': h_t0,
                                    'h_X_out': h_X_out, 'h_Y_out': h_Y_out}.items():
            print(f'{buf_name}: {buf_value.nbytes / 1e6} MB')

    buf_time = time.time()

    # Create the input arrays in device memory and copy data from host
    d_field_x, d_field_y, d_field_t, d_field_U, d_field_V, d_x0, d_y0, d_t0 = \
        (cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)
         for hostbuf in (h_field_x, h_field_y, h_field_t, h_field_U, h_field_V, h_x0, h_y0, h_t0))
    # Create the output arrays in device memory
    d_X_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_X_out.nbytes)
    d_Y_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_Y_out.nbytes)
    buf_time = time.time() - buf_time

    # CHECK KERNEL ARGUMENTS
    check_args(h_field_x, np.uint32(len(h_field_x)),
               h_field_y, np.uint32(len(h_field_y)),
               h_field_t, np.uint32(len(h_field_t)),
               h_field_U, h_field_V,
               h_x0, h_y0, h_t0,
               np.float32(dt), np.uint32(num_timesteps), np.uint32(save_every),
               h_X_out, h_Y_out)

    # Execute the kernel over the entire range of our 1d input
    # allowing OpenCL runtime to select the work group items for the device

    advect = program.advect
    advect.set_scalar_arg_dtypes([None, np.uint32, None, np.uint32, None, np.uint32,
                                  None, None,
                                  None, None, None,
                                  np.float32, np.uint32, np.uint32,
                                  None, None])
    kernel_time = time.time()
    advect(queue, (num_particles,), None,
           d_field_x, np.uint32(len(h_field_x)),
           d_field_y, np.uint32(len(h_field_y)),
           d_field_t, np.uint32(len(h_field_t)),
           d_field_U, d_field_V,
           d_x0, d_y0, d_t0,
           np.float32(dt), np.uint32(num_timesteps), np.uint32(save_every),
           d_X_out, d_Y_out)

    # Wait for the commands to finish before reading back
    queue.finish()
    kernel_time = time.time() - kernel_time

    # Read back the results from the compute device
    tic = time.time()
    cl.enqueue_copy(queue, h_X_out, d_X_out)
    cl.enqueue_copy(queue, h_Y_out, d_Y_out)
    buf_time += time.time() - tic

    # store results in Dataset
    lon = np.concatenate([p0.lon.values[:, np.newaxis],
                          h_X_out.reshape([num_particles, out_timesteps])], axis=1)
    lat = np.concatenate([p0.lat.values[:, np.newaxis],
                          h_Y_out.reshape([num_particles, out_timesteps])], axis=1)
    P = xr.Dataset(data_vars={'lon': (['p_id', 'time'], lon),
                              'lat': (['p_id', 'time'], lat)},
                   coords={'p_id': np.arange(num_particles),
                           'time': advect_time[::save_every]})

    if verbose:
        print(f'memory operations took {buf_time: .3f} seconds')
        print(f'kernel execution took  {kernel_time: .3f} seconds')

    return P, buf_time, kernel_time
