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
    seawater_density: xr.Dataset,
    num_particles: int,
    advect_time: pd.DatetimeIndex,
    save_every: int,
) -> Tuple[
    List[pd.DatetimeIndex], List[xr.Dataset], List[xr.Dataset], List[xr.Dataset]
]:
    """given the parameters for advection, return parameters for an iterative advection"""
    out_time = advect_time[::save_every]
    # each element of out_time marks a time at which the driver will return particle position

    # estimate total size of memory we need to eventually run through the device
    field_bytes, output_bytes, p0_bytes = estimate_memory_bytes(
        current=current,
        wind=wind,
        seawater_density=seawater_density,
        num_particles=num_particles,
        out_timesteps=len(out_time) - 1,
    )

    num_chunks = math.ceil(
        (field_bytes + output_bytes) / (device_bytes - p0_bytes)
    )  # minimum chunking to potentially fit RAM

    if num_chunks > len(current.time):
        raise RuntimeError(
            "Particles take up too much device memory; not enough space for even one field timestep. "
            "Decrease number of particles and try again."
        )

    # it's hard to pre-compute the exact number of chunks necessary that will fit into RAM.  we start with a good guess,
    # but if any of the chunks we create don't fit into RAM, just increment the number of chunks and try again.
    while True:
        # now we split up the advection OUTPUT into chunks.  All else will be based on this splitting.
        assert (
            len(out_time) >= num_chunks
        ), "Cannot split computation, output frequency is too low!"
        # the above situation arises when the span of time between particle save points corresponds to a chunk of field
        # which is too large to fit onto the compute device.

        out_time_chunks = np.array_split(out_time, num_chunks)
        out_time_chunks[1:] = [
            out_time_chunks[i - 1][-1:].append(out_time_chunks[i])
            for i in range(1, len(out_time_chunks))
        ]
        # give subsequent time chunks overlapping endpoints.  This is because the final reported value
        # from a computation will be fed to the next computation as the start point, at the same time.

        advect_time_chunks = [
            advect_time[
                (out_time_chunk[0] <= advect_time) & (advect_time <= out_time_chunk[-1])
            ]
            for out_time_chunk in out_time_chunks
        ]
        current_chunks = chunk_dataset(current, out_time_chunks)
        wind_chunks = chunk_dataset(wind, out_time_chunks)
        seawater_density_chunks = chunk_dataset(seawater_density, out_time_chunks)

        if all(
            sum(
                estimate_memory_bytes(
                    current=current_chunk,
                    wind=wind_chunk,
                    seawater_density=seawater_density_chunk,
                    num_particles=num_particles,
                    out_timesteps=len(out_time_chunk) - 1,
                )
            )
            < device_bytes
            for current_chunk, wind_chunk, seawater_density_chunk, out_time_chunk in zip(
                current_chunks, wind_chunks, seawater_density_chunks, out_time_chunks
            )
        ):
            break
        num_chunks += 1

    return advect_time_chunks, current_chunks, wind_chunks, seawater_density_chunks


def estimate_memory_bytes(
    current: xr.Dataset,
    wind: xr.Dataset,
    seawater_density: xr.Dataset,
    num_particles: int,
    out_timesteps: int,
) -> Tuple[int, int, int]:
    """This estimates total memory needed for the buffers.
    There's a bit more needed for the scalar arguments, but this is tiny"""
    current_bytes = 3 * 4 * np.prod(
        current.U.shape, dtype=np.int64
    ) + 8 * (  # three 32-bit fields
        len(current.lon) + len(current.lat) + len(current.depth) + len(current.time)
    )  # the 4 64-bit coordinate arrays
    wind_bytes = 2 * 4 * np.prod(
        wind.U.shape, dtype=np.int64
    ) + 8 * (  # two 32-bit fields
        len(wind.lon) + len(wind.lat) + len(wind.time)
    )  # the 3 64-bit coordinate arrays
    seawater_density_bytes = 1 * 4 * np.prod(
        seawater_density.rho.shape, dtype=np.int64
    ) + 8 * (
        len(seawater_density.lon)
        + len(seawater_density.lat)
        + len(seawater_density.depth)
        + len(seawater_density.time)
    )
    output_bytes = (
        3 * 4 * num_particles * out_timesteps
        + 1  # three 32-bit variables for each particle for each timestep
        * num_particles
    )  # one byte holding error code for each particle
    p0_bytes = 3 * 4 * num_particles  # three 32-bit variables for each particle
    return (current_bytes + wind_bytes + seawater_density_bytes), output_bytes, p0_bytes


def chunk_dataset(
    dataset: xr.Dataset, out_time_chunks: List[pd.DatetimeIndex]
) -> List[xr.Dataset]:
    """
    Returns a list of slices from dataset, corresponding to the data needed for each chunk in out_time_chunks
    :param dataset: xarray Dataset with a "time" coordinate
    :param out_time_chunks: a sequence of timestamps which each define the output times of a computation
    :return:
    """
    dataset_chunks = []
    for out_time in out_time_chunks:
        start_time = dataset.time.sel(time=out_time[0], method="nearest")
        end_time = dataset.time.sel(time=out_time[-1], method="nearest")
        dataset_chunks.append(
            dataset.sel(time=slice(start_time, end_time))
        )  # this slice is guaranteed to be at least length 1
    return dataset_chunks
