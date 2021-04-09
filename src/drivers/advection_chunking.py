"""
Utilities which help split up advection based on memory constraints
"""
import pandas as pd
import xarray as xr
import math
import numpy as np

from typing import List, Tuple
from tqdm import tqdm


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

    # it's hard to pre-compute the exact number of chunks necessary that will fit into RAM.  we start with a guess,
    # then if any of the chunks we create don't fit into RAM, we just increment the number of chunks and try again.
    print("\tIncrementing number of chunks until data fits...")
    num_chunks = math.ceil((field_bytes + output_bytes) / (device_bytes - p0_bytes))
    pbar = tqdm(total=len(out_time)-num_chunks)
    while True:
        if num_chunks > len(out_time):
            raise RuntimeError(
                "There is not enough memory to hold even the smallest possible chunk!"
            )
        # now we split up the advection OUTPUT into chunks.  All else will be based on this splitting.
        out_time_chunks = np.array_split(out_time, num_chunks)
        # give subsequent time chunks overlapping endpoints.  This is because the final reported value
        # from a computation will be fed to the next computation as the start point.
        out_time_chunks[1:] = [
            out_time_chunks[i - 1][-1:].append(out_time_chunks[i])
            for i in range(1, len(out_time_chunks))
        ]

        # now, we split up the datasets.  If any chunk is too big, we
        # immediately quit, increment num_chunks, and try again.
        advect_time_chunks = []
        current_chunks = []
        wind_chunks = []
        seawater_density_chunks = []
        all_chunks_fit = True
        for out_time_chunk in out_time_chunks:
            # extract dataset chunks for this out_time chunk
            advect_time_chunks.append(
                advect_time[
                    (out_time_chunk[0] <= advect_time)
                    & (advect_time <= out_time_chunk[-1])
                ]
            )
            current_chunks.append(extract_dataset_chunk(current, out_time_chunk))
            wind_chunks.append(extract_dataset_chunk(wind, out_time_chunk))
            seawater_density_chunks.append(
                extract_dataset_chunk(seawater_density, out_time_chunk)
            )

            # if this chunk is too big, try again with more chunks.
            memory_bytes = estimate_memory_bytes(
                current=current_chunks[-1],
                wind=wind_chunks[-1],
                seawater_density=seawater_density_chunks[-1],
                num_particles=num_particles,
                out_timesteps=len(out_time_chunk) - 1,
            )
            if sum(memory_bytes) > device_bytes:
                all_chunks_fit = False
                break
        if all_chunks_fit:
            return (
                advect_time_chunks,
                current_chunks,
                wind_chunks,
                seawater_density_chunks,
            )
        num_chunks += 1
        pbar.update(1)


def estimate_memory_bytes(
    current: xr.Dataset,
    wind: xr.Dataset,
    seawater_density: xr.Dataset,
    num_particles: int,
    out_timesteps: int,
) -> Tuple[int, int, int]:
    """This estimates total memory needed for the buffers.
    There's a bit more needed for the scalar arguments, but this is tiny"""
    # current has 3 4-byte variables, and 8-byte coordinate arrays
    current_bytes = 3 * 4 * math.prod(current.U.shape) + 8 * sum(current.U.shape)
    # wind has 2 4-byte variables, and 8-byte coordinate arrays
    wind_bytes = 2 * 4 * math.prod(wind.U.shape) + 8 * sum(wind.U.shape)
    # seawater density has 1 4-byte variable, and 8-byte coordinate arrays
    seawater_density_bytes = 4 * math.prod(seawater_density.rho.shape) + 8 * sum(
        seawater_density.rho.shape
    )
    # output has three 4-byte variables for each particle per timestep, plus 1-byte exit codes
    output_bytes = 3 * 4 * num_particles * out_timesteps + 1 * num_particles
    # the particle input has 3 4-byte variables and 4 8-byte variables
    p0_bytes = (3 * 4 + 4 * 8) * num_particles
    return (current_bytes + wind_bytes + seawater_density_bytes), output_bytes, p0_bytes


def extract_dataset_chunk(
    dataset: xr.Dataset, out_time: pd.DatetimeIndex
) -> xr.Dataset:
    """
    Returns a list of slices from dataset, corresponding to the data needed for each chunk in out_time_chunks
    :param dataset: xarray Dataset with a "time" coordinate
    :param out_time: sequence of timestamps which define the computation domain;
        we will use their extent to slice dataset to match their time domain.
    :return:
    """
    start_time = dataset.time.sel(time=out_time[0], method="nearest")
    end_time = dataset.time.sel(time=out_time[-1], method="nearest")
    return dataset.sel(
        time=slice(start_time, end_time)
    )  # this slice is guaranteed to be at least length 1
