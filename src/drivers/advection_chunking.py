"""
Utilities which help split up advection based on memory constraints
"""
import pandas as pd
import xarray as xr
import math
import numpy as np

from typing import List, Tuple


def chunk_advection_params(
    device_bytes: int,
    current: xr.Dataset,
    wind: xr.Dataset,
    bathymetry: xr.Dataset,
    num_particles: int,
    advect_time: pd.DatetimeIndex,
    save_every: int
) -> Tuple[List[pd.DatetimeIndex], List[pd.DatetimeIndex], List[xr.Dataset], List[xr.Dataset]]:
    """given the parameters for advection, return parameters for an iterative advection"""
    out_time = advect_time[::save_every]
    # each element of out_time marks a time at which the driver will return particle position

    # estimate total size of memory we need to eventually run through the device
    field_bytes, bathymetry_bytes, output_bytes, p0_bytes = estimate_memory_bytes(
        current=current,
        wind=wind,
        bathymetry=bathymetry,
        num_particles=num_particles,
        out_timesteps=len(out_time) - 1
    )

    num_chunks = math.ceil((field_bytes + output_bytes) / (device_bytes - p0_bytes - bathymetry_bytes))  # minimum chunking to potentially fit RAM
    if num_chunks > len(current.time):
        raise RuntimeError('Particles take up too much device memory; not enough space for even one field timestep. '
                           'Decrease number of particles and try again.')

    # it's hard to pre-compute the exact number of chunks necessary that will fit into RAM.  we start with a good guess,
    # but if any of the chunks we create don't fit into RAM, just increment the number of chunks and try again.
    while True:
        # now we split up the advection OUTPUT into chunks.  All else will be based on this splitting.
        assert len(out_time) >= num_chunks, 'Cannot split computation, output frequency is too low!'
        # the above situation arises when the span of time between particle save points corresponds to a chunk of field
        # which is too large to fit onto the compute device.

        out_time_chunks = np.array_split(out_time, num_chunks)
        out_time_chunks[1:] = [out_time_chunks[i-1][-1:].append(out_time_chunks[i])
                               for i in range(1, len(out_time_chunks))]
        # give subsequent time chunks overlapping endpoints.  This is because the final reported value
        # from a computation will be fed to the next computation as the start point, at the same time.

        advect_time_chunks = [advect_time[(out_time_chunk[0] <= advect_time) & (advect_time <= out_time_chunk[-1])]
                              for out_time_chunk in out_time_chunks]

        current_chunks = [current.sel(time=slice(out_time_chunk[0], out_time_chunk[-1]))
                          for out_time_chunk in out_time_chunks]
        wind_chunks = [wind.sel(time=slice(out_time_chunk[0], out_time_chunk[-1]))
                       for out_time_chunk in out_time_chunks]

        if all(sum(estimate_memory_bytes(current=current_chunk,
                                         wind=wind_chunk,
                                         bathymetry=bathymetry,
                                         num_particles=num_particles,
                                         out_timesteps=len(out_time_chunk)-1)) < device_bytes
               for current_chunk, wind_chunk, out_time_chunk in zip(current_chunks, wind_chunks, out_time_chunks)):
            break
        num_chunks += 1

    return advect_time_chunks, out_time_chunks, current_chunks, wind_chunks


def estimate_memory_bytes(current: xr.Dataset,
                          wind: xr.Dataset,
                          bathymetry: xr.Dataset,
                          num_particles: int,
                          out_timesteps: int,
                          ) -> Tuple[int, int, int, int]:
    """This estimates total memory needed for the buffers.
    There's a bit more needed for the scalar arguments, but this is tiny"""
    current_bytes = (3 * 4 * np.prod(current.U.shape) +  # three 32-bit fields
                     8 * (len(current.lon) + len(current.lat) + len(current.depth) + len(current.time)))  # the 4 64-bit coordinate arrays
    wind_bytes = (2 * 4 * np.prod(wind.U.shape) +  # two 32-bit fields
                  8 * (len(wind.lon) + len(wind.lat) + len(wind.time)))  # the 3 64-bit coordinate arrays
    bathymetry_bytes = (1 * 4 * np.prod(bathymetry.elevation.shape) +  # 1 32-bit field
                        8 * (len(bathymetry.lon) + len(bathymetry.lat)))  # 2 64-bit coordinate arrays
    output_bytes = (3 * 4 * num_particles * out_timesteps +  # three 32-bit variables for each particle for each timestep
                    1 * num_particles)  # one byte holding error code for each particle
    p0_bytes = 3 * 4 * num_particles  # three 32-bit variables for each particle
    return (current_bytes+wind_bytes), bathymetry_bytes, output_bytes, p0_bytes
