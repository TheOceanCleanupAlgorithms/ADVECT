import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd

from typing import Tuple
from kernels.kernels import EulerianKernel2D


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
    field = field.transpose('time', 'lon', 'lat')  # make sure the underlying numpy arrays are in the correct shape

    # calculate constants associated with advection
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

    # perform the basic steps of the advection calculation, leaving details up to subfunctions
    kernel = create_kernel(context, field=field, p0=p0, num_particles=num_particles,
                           dt=dt, t0=t0, num_timesteps=num_timesteps, save_every=save_every,
                           out_timesteps=out_timesteps)

    X_out, Y_out = kernel.execute_and_return_result()

    # store results in Dataset
    P = create_dataset_from_advection_output(p0=p0, X_out=X_out, Y_out=Y_out,
                                             num_particles=num_particles, out_timesteps=out_timesteps,
                                             advect_time=advect_time, save_every=save_every)

    if verbose:
        kernel.print_memory_footprint()
        kernel.print_execution_time()

    return P, kernel.buf_time, kernel.kernel_time


def create_kernel(context: cl.Context, field: xr.Dataset, p0: pd.DataFrame, num_particles: int,
                  dt: float, t0: pd.Timestamp, num_timesteps: int, save_every: int, out_timesteps: int):
    """create and return the wrapper for the opencl kernel"""
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


def create_dataset_from_advection_output(p0, X_out, Y_out, num_particles, out_timesteps, advect_time, save_every):
    lon = np.concatenate([p0.lon.values[:, np.newaxis],
                          X_out.reshape([num_particles, out_timesteps])], axis=1)
    lat = np.concatenate([p0.lat.values[:, np.newaxis],
                          Y_out.reshape([num_particles, out_timesteps])], axis=1)
    P = xr.Dataset(data_vars={'lon': (['p_id', 'time'], lon),
                              'lat': (['p_id', 'time'], lat)},
                   coords={'p_id': np.arange(num_particles),
                           'time': advect_time[::save_every]})
    return P
