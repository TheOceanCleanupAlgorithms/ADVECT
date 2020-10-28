"""
Utilities which help split up advection based on memory constraints
"""
import pandas as pd
import xarray as xr
import math
import numpy as np

from typing import List, Tuple


def chunk_advection_params(device_bytes: int, field: xr.Dataset, num_particles: int, advect_time: pd.DatetimeIndex,
                           save_every: int) -> Tuple[List[pd.DatetimeIndex], List[pd.DatetimeIndex], List[xr.Dataset]]:
    """given the parameters for advection, return parameters for an iterative advection"""
    out_time = advect_time[::save_every]
    # each element of out_time marks a time at which the driver will return particle position

    # estimate total size of memory we need to eventually run through the device
    field_bytes, output_bytes, p0_bytes = estimate_memory_bytes(field.sel(time=slice(advect_time[0], advect_time[-1])),
                                                                num_particles, len(out_time)-1)
    available_bytes_for_field = device_bytes - (output_bytes + p0_bytes)
    if available_bytes_for_field <= 0:
        raise RuntimeError('Particles take up all of device memory; no space for field chunks. '
                           'Decrease number of particles and try again.')
    num_chunks = math.ceil(field_bytes / available_bytes_for_field)  # minimum chunking to potentially fit RAM
    if num_chunks > len(field.time):
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

        field_chunks = [field.sel(time=slice(out_time_chunk[0], out_time_chunk[-1]))
                        for out_time_chunk in out_time_chunks]

        if all(sum(estimate_memory_bytes(field_chunk, num_particles, len(out_time_chunk)-1)) < device_bytes
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
