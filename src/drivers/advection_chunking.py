"""
Utilities which help split up advection based on memory constraints
"""
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from enums.forcings import Forcing


def chunk_advection_params(
    device_bytes: int,
    forcing_data: Dict[Forcing, xr.Dataset],
    num_particles: int,
    advect_time: pd.DatetimeIndex,
    save_every: int,
) -> Tuple[List[pd.DatetimeIndex], List[Dict[Forcing, xr.Dataset]]]:
    """given the parameters for advection, return parameters for an iterative advection"""
    out_time = advect_time[::save_every]
    # each element of out_time marks a time at which the driver will return particle position

    # estimate total size of memory we need to eventually run through the device
    field_bytes, output_bytes, p0_bytes = estimate_memory_bytes(
        forcing_data=forcing_data,
        num_particles=num_particles,
        out_timesteps=len(out_time) - 1,
    )

    # it's hard to pre-compute the exact number of chunks necessary that will fit into RAM.  we start with a guess,
    # then if any of the chunks we create don't fit into RAM, we just increment the number of chunks and try again.
    print("\tIncrementing number of chunks until data fits...")
    num_chunks = 1
    with tqdm(total=len(out_time) - num_chunks) as pbar:
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
            forcing_data_chunks = []
            all_chunks_fit = True
            for out_time_chunk in out_time_chunks:
                # extract dataset chunks for this out_time chunk
                advect_time_chunks.append(
                    advect_time[
                        (out_time_chunk[0] <= advect_time)
                        & (advect_time <= out_time_chunk[-1])
                    ]
                )
                forcing_data_chunks.append(
                    {
                        forcing: extract_dataset_chunk(ds, out_time_chunk)
                        for forcing, ds in forcing_data.items()
                    }
                )

                # if this chunk is too big, try again with more chunks.
                memory_bytes = estimate_memory_bytes(
                    forcing_data=forcing_data_chunks[-1],
                    num_particles=num_particles,
                    out_timesteps=len(out_time_chunk) - 1,
                )
                if sum(memory_bytes) > device_bytes:
                    all_chunks_fit = False
                    break
            if all_chunks_fit:
                return advect_time_chunks, forcing_data_chunks
            num_chunks += 1
            pbar.update(1)


def estimate_memory_bytes(
    forcing_data: Dict[Forcing, xr.Dataset],
    num_particles: int,
    out_timesteps: int,
) -> Tuple[int, int, int]:
    """This estimates total memory needed for the buffers.
    There's a bit more needed for the scalar arguments, but this is tiny"""
    forcing_data_bytes = 0
    for ds in forcing_data.values():
        field_shape = list(ds.data_vars.values())[0].shape
        # when passed to the kernel, field variables are represented as float32, coordinates as float64
        forcing_data_bytes += len(ds.data_vars) * 4 * math.prod(field_shape) + 8 * sum(
            field_shape
        )

    # output has three 4-byte variables for each particle per timestep, plus 1-byte exit codes
    output_bytes = 3 * 4 * num_particles * out_timesteps + 1 * num_particles
    # the particle input has 3 4-byte variables and 4 8-byte variables
    p0_bytes = (3 * 4 + 4 * 8) * num_particles
    return forcing_data_bytes, output_bytes, p0_bytes


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
