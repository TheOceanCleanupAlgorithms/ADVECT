import datetime
import gc
import logging
import time
import pyopencl as cl
import numpy as np
import xarray as xr
import pandas as pd

from tqdm import tqdm
from typing import Tuple, Optional, List, Type
from pathlib import Path
from drivers.advection_chunking import chunk_advection_params
from drivers.create_bathymetry import create_bathymetry
from enums.forcings import Forcing
from io_tools.OutputWriter import OutputWriter
from kernel_wrappers.Kernel import Kernel
from kernel_wrappers.Kernel3D import Kernel3D, AdvectionScheme
from kernel_wrappers.kernel_constants import EXIT_CODES


def openCL_advect(
    forcing_data: dict[Forcing, xr.Dataset],
    kernel_cls: Type[Kernel],
    output_writer: OutputWriter,
    p0: xr.Dataset,
    start_time: datetime.datetime,
    dt: datetime.timedelta,
    num_timesteps: int,
    save_every: int,
    advection_scheme: AdvectionScheme,
    eddy_diffusivity: xr.Dataset,
    max_wave_height: float,
    wave_mixing_depth_factor: float,
    windage_multiplier: Optional[float],
    wind_mixing_enabled: bool,
    memory_utilization: float,
    platform_and_device: Tuple[int] = None,
) -> List[Path]:
    """
    Splits an advection computation into chunks, based on memory constraints of compute device,
    and executes the kernel over each chunk.
    :param forcing_data: dictionary holding whatever forcing data is available.
        valid keys: {"current", "wind", "seawater_density"}
    :param kernel_cls: kernel class used to execute the chunks
    :param output_writer: object which is responsible for persisting the model output to disk
    :param p0: xarray Dataset storing particle initial state from sourcefile
    :param start_time: advection start time
    :param dt: timestep duration
    :param num_timesteps: number of timesteps
    :param save_every: how many timesteps between saving state.  Must divide num_timesteps.
    :param advection_scheme: scheme to use, listed in the AdvectionScheme enum
    :param eddy_diffusivity: xarray Dataset storing vertical profiles of eddy diffusivities
    :param max_wave_height: caps parameterization in kernel; see config_specifications.md
    :param wave_mixing_depth_factor: scales depth of mixing in kernel; see config_specifications.md
    :param windage_multiplier: multiplies the default windage, which is based on emerged area
    :param wind_mixing_enabled: toggle the wind mixing functionality
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

    # encode the model domain, taken as where all the current components are non-null, as bathymetry
    print("Calculating bathymetry of current dataset...")
    forcing_data[Forcing.current] = xr.merge(
        (
            forcing_data[Forcing.current],
            create_bathymetry(forcing_data[Forcing.current]),
        )
    )
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
            advection_scheme=advection_scheme,
            config={
                "eddy_diffusivity": eddy_diffusivity,
                "max_wave_height": max_wave_height,
                "wave_mixing_depth_factor": wave_mixing_depth_factor,
                "windage_multiplier": windage_multiplier,
                "wind_mixing_enabled": wind_mixing_enabled,
            },
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
        p0_chunk = kernel.get_final_state()

        del kernel  # important for releasing memory for the next iteration
        gc.collect()

        print("\tWriting output to disk...")
        output_start = time.time()
        output_writer.write_output_chunk(P_chunk)
        output_time = time.time() - output_start

        print("\t---BUFFER SIZES---")
        print(f'\tCurrent:            {memory_usage["current"] / 1e6:10.3f} MB')
        print(f'\tWind:               {memory_usage["wind"] / 1e6:10.3f} MB')
        print(f'\tSeawater Density:   {memory_usage["seawater_density"] / 1e6:10.3f} MB')
        print(f'\tParticle State:     {memory_usage["particles"] / 1e6:10.3f} MB')
        print(f'\tTotal:              {sum(memory_usage.values()) / 1e6:10.3f} MB')
        print("\t---EXECUTION TIME---")
        print(f"\tData Loading:      {data_loading_time:10.3f}s")
        print(f"\tBuffer Read/Write: {buffer_time:10.3f}s")
        print(f"\tKernel Execution:  {execution_time:10.3f}s")
        print(f"\tOutput Writing:    {output_time:10.3f}s")

    return output_writer.paths


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
