import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyopencl as cl
import xarray as xr

from enums.advection_scheme import AdvectionScheme
from enums.forcings import Forcing
from kernel_wrappers.Field3D import (
    Field3D,
    create_empty_2d_field,
    is_sorted_ascending,
    buffer_from_array,
)
from kernel_wrappers.Kernel import Kernel, KernelConfig

KERNEL_SOURCE = Path(__file__).parent.parent / "kernels/kernel_3d.cl"
MODEL_CORE = Path(__file__).parent.parent / "model_core"


@dataclass
class Kernel3DConfig(KernelConfig):
    """Configuration for 3D Kernel.
    advection_scheme: which mathematical scheme to use for an advection step
    windage_multiplier: scales the physically-motivated windage contribution
    wind_mixing_enabled: toggles wind-driven mixing
    max_wave_height: (m) see configfile_specifications.md for details
    wave_mixing_depth_factor: see configfile_specifications.md for details
    eddy_diffusivity: dataset with vertical profiles of horizontal/vertical eddy diffusivity
        variables [m^2 / s] (dim [m]):
        horizontal_diffusivity (z_hd)
        vertical_diffusivity (z_vd)
    """

    advection_scheme: AdvectionScheme
    windage_multiplier: Optional[float]
    wind_mixing_enabled: bool
    max_wave_height: float
    wave_mixing_depth_factor: float
    eddy_diffusivity: xr.Dataset


class Kernel3D(Kernel):
    """wrapper for src/kernels/kernel_3d.cl"""

    def __init__(
        self,
        forcing_data: Dict[Forcing, xr.Dataset],
        p0: xr.Dataset,
        advect_time: pd.DatetimeIndex,
        save_every: int,
        config: Kernel3DConfig,
        context: cl.Context,
    ):
        """
        :param forcing_data:
            required keys: {Forcing.current, Forcing.seawater_density}
            optional keys: {Forcing.wind}
        :param p0: initial state of particles
        :param advect_time: the timeseries which the kernel advects on
        :param save_every: number of timesteps between each writing of particle state
        :param config: see Kernel3DConfig definition for details
        :param context: PyopenCL context for executing OpenCL programs
        """
        # save some arguments for creating output dataset
        self.p0 = p0
        self.out_time = advect_time[::save_every][1:]
        self.advect_time = advect_time

        # some handy timers
        self.data_load_time = 0
        self.buf_time = 0
        self.kernel_time = 0

        # ---KERNEL ARGUMENT INITIALIZATION--- #
        # 1-to-1 for arguments in kernel_3d.cl::advect; see comments there for details

        print("\t\tLoading Current Data...")
        data_loading_start = time.time()
        self.current = Field3D(
            ds=forcing_data[Forcing.current], varnames=["U", "V", "W", "bathymetry"]
        )
        # wind vector field
        print("\t\tLoading Wind Data...")
        if "wind" in forcing_data:
            self.wind = Field3D(ds=forcing_data[Forcing.wind], varnames=["U", "V"])
        else:  # Windage disabled; pass a dummy field with singleton dimensions
            self.wind = create_empty_2d_field()
            self.windage_multiplier = (
                np.nan
            )  # to flag the kernel that windage is disabled
        # seawater_density vector field
        print("\t\tLoading Seawater Density Data...")
        self.seawater_density = Field3D(
            ds=forcing_data[Forcing.seawater_density],
            varnames=["rho"],
            non_uniform_time=True,
        )
        self.data_load_time = time.time() - data_loading_start
        # particle initialization
        self.x0 = p0.lon.values.astype(np.float32)
        self.y0 = p0.lat.values.astype(np.float32)
        self.z0 = p0.depth.values.astype(np.float32)
        self.release_date = p0.release_date.values.astype("datetime64[s]").astype(
            np.float64
        )
        self.radius = p0.radius.values.astype(np.float64)
        self.density = p0.density.values.astype(np.float64)
        self.corey_shape_factor = p0.corey_shape_factor.values.astype(np.float64)
        # advection time parameters
        self.start_time = np.float64(advect_time[0].timestamp())
        self.dt = np.float64(pd.Timedelta(advect_time.freq).total_seconds())
        self.ntimesteps = np.uint32(
            len(advect_time) - 1
        )  # minus one bc initial position given
        self.save_every = np.uint32(save_every)
        # output_vectors (initialized to nan)
        out_shape = len(p0.lon) * len(self.out_time)
        self.X_out = np.full(out_shape, np.nan, dtype=np.float32)
        self.Y_out = np.full(out_shape, np.nan, dtype=np.float32)
        self.Z_out = np.full(out_shape, np.nan, dtype=np.float32)
        # physics
        self.advection_scheme = config.advection_scheme.value
        self.windage_multiplier = config.windage_multiplier
        self.wind_mixing_enabled = config.wind_mixing_enabled
        self.max_wave_height = config.max_wave_height
        self.wave_mixing_depth_factor = config.wave_mixing_depth_factor
        # eddy diffusivity
        self.horizontal_eddy_diffusivity_z = config.eddy_diffusivity.z_hd.values.astype(
            np.float64
        )
        self.horizontal_eddy_diffusivity_values = (
            config.eddy_diffusivity.horizontal_diffusivity.values.astype(np.float64)
        )
        self.vertical_eddy_diffusivity_z = config.eddy_diffusivity.z_vd.values.astype(
            np.float64
        )
        self.vertical_eddy_diffusivity_values = (
            config.eddy_diffusivity.vertical_diffusivity.values.astype(np.float64)
        )
        # debugging
        self.exit_code = p0.exit_code.values.astype(np.byte)

        # ---HOST INITIALIZATIONS--- #
        # create opencl objects
        self.context = context
        self.queue = cl.CommandQueue(context)
        self.cl_kernel = (
            cl.Program(context, open(KERNEL_SOURCE).read())
            .build(options=["-I", str(MODEL_CORE)])
            .advect
        )

        self.execution_result = None

    def execute(self) -> xr.Dataset:
        """tranfers arguments to the compute device, triggers execution, returns result"""
        # perform argument check
        self._check_args()

        # write arguments to compute device
        print("\t\tWriting buffers to compute device...")
        write_start = time.time()
        current_args = self.current.create_kernel_arguments(context=self.context)
        wind_args = self.wind.create_kernel_arguments(context=self.context)
        seawater_density_args = self.seawater_density.create_kernel_arguments(
            context=self.context
        )
        p0_buffers = [
            buffer_from_array(arr=arr, context=self.context)
            for arr in (
                self.x0,
                self.y0,
                self.z0,
                self.release_date,
                self.radius,
                self.density,
                self.corey_shape_factor,
            )
        ]
        eddy_diffusity_buffers = [
            buffer_from_array(arr=arr, context=self.context)
            for arr in (
                self.horizontal_eddy_diffusivity_z,
                self.horizontal_eddy_diffusivity_values,
                self.vertical_eddy_diffusivity_z,
                self.vertical_eddy_diffusivity_values,
            )
        ]

        out_arrays = (self.X_out, self.Y_out, self.Z_out, self.exit_code)
        out_buffers = [
            cl.Buffer(
                self.context,
                cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=arr,
            )
            for arr in out_arrays
        ]
        self.buf_time = time.time() - write_start

        # execute the program
        print("\t\tExecuting kernel...")
        execution_start = time.time()
        self.cl_kernel(
            self.queue,
            (len(self.x0),),
            None,
            *current_args,
            *wind_args,
            *seawater_density_args,
            *p0_buffers,
            np.uint32(self.advection_scheme),
            np.float64(self.windage_multiplier),
            np.uint32(self.wind_mixing_enabled),
            np.float64(self.max_wave_height),
            np.float64(self.wave_mixing_depth_factor),
            *eddy_diffusity_buffers,
            np.uint32(len(self.horizontal_eddy_diffusivity_values)),
            np.uint32(len(self.vertical_eddy_diffusivity_values)),
            self.start_time,
            self.dt,
            self.ntimesteps,
            self.save_every,
            *out_buffers,
        )

        # wait for the computation to complete
        self.queue.finish()
        self.kernel_time = time.time() - execution_start

        # Read back the results from the compute device
        print("\t\tCopying results back to host device...")
        read_start = time.time()
        for arr, buffer in zip(out_arrays, out_buffers):
            cl.enqueue_copy(self.queue, arr, buffer)
        self.buf_time += time.time() - read_start

        # create and return dataset
        P = self.p0.assign_coords({"time": self.out_time})  # add a time dimension
        self.execution_result = P.assign(  # overwrite with new data
            {
                "lon": (["p_id", "time"], self.X_out.reshape([len(P.p_id), -1])),
                "lat": (["p_id", "time"], self.Y_out.reshape([len(P.p_id), -1])),
                "depth": (["p_id", "time"], self.Z_out.reshape([len(P.p_id), -1])),
                "exit_code": (["p_id"], self.exit_code),
            }
        )
        return self.execution_result

    def get_memory_footprint(self) -> dict:
        particle_bytes = (
            self.x0.nbytes
            + self.y0.nbytes
            + self.z0.nbytes
            + self.release_date.nbytes
            + self.density.nbytes
            + self.radius.nbytes
            + self.corey_shape_factor.nbytes
            + self.X_out.nbytes
            + self.Y_out.nbytes
            + self.Z_out.nbytes
            + self.exit_code.nbytes
        )
        return {
            "current": self.current.memory_usage_bytes(),
            "wind": self.wind.memory_usage_bytes(),
            "seawater_density": self.seawater_density.memory_usage_bytes(),
            "particles": particle_bytes,
        }

    def get_data_loading_time(self) -> float:
        return self.data_load_time

    def get_buffer_transfer_time(self) -> float:
        return self.buf_time

    def get_kernel_execution_time(self) -> float:
        return self.kernel_time

    def _check_args(self):
        """ensure kernel arguments satisfy constraints"""

        # check eddy diffusion valid
        assert is_sorted_ascending(self.horizontal_eddy_diffusivity_z)
        assert is_sorted_ascending(self.vertical_eddy_diffusivity_z)

        # check particle positions valid
        assert np.nanmax(self.x0) < 180
        assert np.nanmin(self.x0) >= -180
        assert np.nanmax(self.y0) <= 90
        assert np.nanmin(self.y0) >= -90
        assert np.nanmax(self.z0) <= 0

        # check corey shape factor valid
        assert np.all((0.15 < self.corey_shape_factor) & (self.corey_shape_factor <= 1))

        # check enum valid
        assert self.advection_scheme in (0, 1)

        # issue warning if wind timestep is smaller than one day
        if np.any(
            np.diff(self.wind.coords["t"]) < pd.Timedelta(days=1).total_seconds()
        ):
            warnings.warn(
                "Timestep of wind data is less than a day.  The kernel assumes a fully developed sea state from each "
                "wind datum; short timesteps mean this is a bad assumption.  Use wind data averaged over a longer "
                "timestep, or complain to the developers (or both)."
            )
