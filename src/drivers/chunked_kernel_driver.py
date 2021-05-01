import datetime
import gc
import logging
import time
import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd

from tqdm import tqdm
from typing import Dict, Tuple, List, Type
from pathlib import Path
from drivers.advection_chunking import chunk_advection_params
from enums.forcings import Forcing
from io_tools.OutputWriter import OutputWriter
from kernel_wrappers.Kernel import Kernel, KernelConfig
from kernel_wrappers.kernel_constants import EXIT_CODES


def execute_chunked_kernel_computation(
    forcing_data: Dict[Forcing, xr.Dataset],
    kernel_cls: Type[Kernel],
    kernel_config: KernelConfig,
    output_writer: OutputWriter,
    p0: xr.Dataset,
    start_time: datetime.datetime,
    dt: datetime.timedelta,
    num_timesteps: int,
    save_every: int,
    memory_utilization: float,
    platform_and_device: Tuple[int] = None,
) -> List[Path]:
    """
    Splits an advection computation into chunks, based on memory constraints of compute device,
    and executes the kernel over each chunk.
    :param forcing_data: dictionary holding whatever forcing data is available.
        valid keys: {"current", "wind", "seawater_density"}
    :param kernel_cls: kernel class used to execute the chunks
    :param kernel_config: dictionary to be passed to kernel_cls.__init__ as arg "config."
        See kernel_cls implementation for details on what configuration parameters are required.
    :param output_writer: object which is responsible for persisting the model output to disk
    :param p0: xarray Dataset storing particle initial state from sourcefile
    :param start_time: advection start time
    :param dt: timestep duration
    :param num_timesteps: number of timesteps
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param memory_utilization: fraction of the opencl device memory available for buffers
    :param platform_and_device: indices of platform/device to execute program.  None initiates interactive mode.
    :return: list of outputfile paths
    """
    num_particles = len(p0.p_id)
    advect_time = pd.date_range(start=start_time, freq=dt, periods=num_timesteps)
    # choose the device/platform we're running on
    if platform_and_device is None:
        context = cl.create_some_context(interactive=True)
    else:
        context = cl.create_some_context(answers=list(platform_and_device))

    # get the minimum RAM available on the specified compute devices.
    print("Chunking Datasets...")
    available_RAM = min(device.global_mem_size for device in context.devices) * memory_utilization
    advect_time_chunks, forcing_data_chunks = chunk_advection_params(
        device_bytes=available_RAM,
        forcing_data=forcing_data,
        num_particles=num_particles,
        advect_time=advect_time,
        save_every=save_every,
    )

    create_logger(output_writer.folder_path / "warnings.log")
    p0_chunk = p0.assign({'exit_code': ('p_id', np.zeros(len(p0.p_id)))})
    for i in tqdm(
        range(len(advect_time_chunks)),
        desc="PROGRESS",
        unit="chunk",
    ):
        print(f'Advecting from {advect_time_chunks[i][0]} to {advect_time_chunks[i][-1]}...')
        # create the kernel wrapper object, pass it arguments
        print("\tInitializing Kernel...")
        kernel = kernel_cls(
            forcing_data=forcing_data_chunks[i],
            p0=p0_chunk,
            config=kernel_config,
            advect_time=advect_time_chunks[i],
            save_every=save_every,
            context=context,
        )

        print("\tTriggering kernel execution...")
        P_chunk = kernel.execute()
        handle_errors(chunk=P_chunk, chunk_num=i + 1)
        data_loading_time = kernel.get_data_loading_time()
        buffer_time = kernel.get_buffer_transfer_time()
        execution_time = kernel.get_kernel_execution_time()
        memory_usage = kernel.get_memory_footprint()

        del kernel  # important for releasing memory for the next iteration
        gc.collect()

        print("\tWriting output to disk...")
        output_start = time.time()
        output_writer.write_output_chunk(P_chunk)
        output_time = time.time() - output_start

        p0_chunk = convert_final_state_to_initial_state(
                execution_result=P_chunk,
                advect_time=advect_time_chunks[i],
                previous_initial_state=p0_chunk,
        )

        print("\t---BUFFER SIZES---")
        for key, value in memory_usage.items():
            print(f'\t{key+":":<20}{value / 1e6:10.3f} MB')
        print(f'\t{"Total:":<20}{sum(memory_usage.values()) / 1e6:10.3f} MB')
        print("\t---EXECUTION TIME---")
        print(f"\tData Loading:         {data_loading_time:10.3f}s")
        print(f"\tBuffer Read/Write:    {buffer_time:10.3f}s")
        print(f"\tKernel Execution:     {execution_time:10.3f}s")
        print(f"\tOutput Writing:       {output_time:10.3f}s")

    return output_writer.paths


def convert_final_state_to_initial_state(
    execution_result: xr.Dataset,
    previous_initial_state: xr.Dataset,
    advect_time: pd.DatetimeIndex
) -> xr.Dataset:
    """Takes the final timestep of a computation, and convert it into an initial state to send to another kernel"""
    final_state = execution_result.isel(time=-1).copy(deep=True)  # copy in order to avoid mutating execution result
    # problem is, this ^ has nans for location of all the unreleased particles.  Restore that information here
    unreleased = final_state.release_date > advect_time[-2]
    for var in execution_result.data_vars:
        if "time" in execution_result[var].dims:
            final_state[var].loc[unreleased] = previous_initial_state[var].loc[unreleased]
    return final_state


def create_logger(log_path: Path):
    """this sets up logging such that logs with level WARNING go to log_path,
        logs with level ERROR or greater go to log_path and stdout."""
    logging.basicConfig(filename=str(log_path), filemode='w', level=logging.WARNING,
                        format="%(asctime)s %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(console)


def handle_errors(chunk: xr.Dataset, chunk_num: int):
    if not np.all(chunk.exit_code == 0):
        bad_codes = np.unique(chunk.exit_code[chunk.exit_code != 0])
        logging.error(f"Error: {np.count_nonzero(chunk.exit_code)} particle(s) did not exit successfully: "
                      f"exit code(s) {[f'{code} ({EXIT_CODES[code]})' for code in bad_codes]}")
        for code in chunk.exit_code[chunk.exit_code != 0]:
            logging.warning(f"Chunk {chunk_num: 3}: Particle ID {int(code.p_id)} exited with error code {int(code)}.")
    if np.any(chunk.exit_code < 0):
        raise ValueError(f"Fatal error encountered, error code(s) "
                         f"{np.unique(chunk.exit_code[chunk.exit_code < 0])}; aborting")
