import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyopencl as cl
import xarray as xr

import kernel_wrappers.kernel_constants as cl_const
from enums.advection_scheme import AdvectionScheme
from enums.forcings import Forcing
from kernel_wrappers.Kernel import Kernel, KernelConfig

KERNEL_SOURCE = Path(__file__).parent.parent / "kernels/kernel_2d.cl"
MODEL_CORE = Path(__file__).parent.parent / "model_core"


@dataclass
class Kernel2DConfig(KernelConfig):
    """Configuration for 2D Kernel.
    advection_scheme: which mathematical scheme to use for an advection step
    windage_coefficient: fraction of wind speed that is transferred to particles during advection
    eddy_diffusivity: (m^2 / s) controls the scale of each particle's Wiener-process random walk.
    """

    advection_scheme: AdvectionScheme
    windage_coefficient: float
    eddy_diffusivity: float


class Kernel2D(Kernel):
    """wrapper for src/kernels/kernel_2d.cl"""

    def __init__(
        self,
        forcing_data: Dict[Forcing, xr.Dataset],
        p0: xr.Dataset,
        advect_time: pd.DatetimeIndex,
        save_every: int,
        config: Kernel2DConfig,
        context: cl.Context,
    ):
        """
        :param forcing_data:
            required keys: {Forcing.current}
            optional keys: {Forcing.wind}
        :param p0: initial state of particles
        :param advect_time: the timeseries which the kernel advects on
        :param save_every: number of timesteps between each writing of particle state
        :param config: see Kernel2DConfig definition for details
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
        current = forcing_data[Forcing.current].transpose(
            "time", "lat", "lon"
        )  # coerce values into correct shape before flattening
        self.current_x = current.lon.values.astype(np.float64)
        self.current_y = current.lat.values.astype(np.float64)
        self.current_t = current.time.values.astype("datetime64[s]").astype(
            np.float64
        )  # float64 representation of unix timestamp
        self.current_U = current.U.values.astype(
            np.float32, copy=False
        ).ravel()  # astype will still copy if variable is not already float32
        self.current_V = current.V.values.astype(np.float32, copy=False).ravel()
        # wind vector field
        print("\t\tLoading Wind Data...")
        if "wind" in forcing_data:
            wind = forcing_data[Forcing.wind].transpose(
                "time", "lat", "lon"
            )  # coerce values into correct shape before flattening
            self.wind_x = wind.lon.values.astype(np.float64)
            self.wind_y = wind.lat.values.astype(np.float64)
            self.wind_t = wind.time.values.astype("datetime64[s]").astype(
                np.float64
            )  # float64 representation of unix timestamp
            self.wind_U = wind.U.values.astype(
                np.float32, copy=False
            ).ravel()  # astype will still copy if variable is not already float32
            self.wind_V = wind.V.values.astype(np.float32, copy=False).ravel()
        else:  # Windage disabled; pass a dummy field with singleton dimensions
            self.wind_x, self.wind_y, self.wind_t = [np.zeros(1, dtype=np.float64)] * 3
            self.wind_U, self.wind_V = [np.zeros((1, 1, 1), dtype=np.float32)] * 2
            self.windage_coefficient = (
                np.nan
            )  # to flag the kernel that windage is disabled
        self.data_load_time = time.time() - data_loading_start
        # particle initialization
        self.x0 = p0.lon.values.astype(np.float32)
        self.y0 = p0.lat.values.astype(np.float32)
        self.release_date = p0.release_date.values.astype("datetime64[s]").astype(
            np.float64
        )
        # advection time parameters
        self.start_time = np.float64(advect_time[0].timestamp())
        self.dt = np.float64(pd.Timedelta(advect_time.freq).total_seconds())
        self.ntimesteps = np.uint32(
            len(advect_time) - 1
        )  # minus one bc initial position given
        self.save_every = np.uint32(save_every)
        # output_vectors
        self.X_out = np.full(
            (len(p0.lon) * len(self.out_time)), np.nan, dtype=np.float32
        )  # output will have this value
        self.Y_out = np.full(
            (len(p0.lat) * len(self.out_time)), np.nan, dtype=np.float32
        )  # until overwritten (e.g. pre-release)
        # physics
        self.advection_scheme = config.advection_scheme.value
        self.windage_coefficient = config.windage_coefficient
        self.eddy_diffusivity = config.eddy_diffusivity
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
        (
            d_current_x,
            d_current_y,
            d_current_t,
            d_current_U,
            d_current_V,
            d_wind_x,
            d_wind_y,
            d_wind_t,
            d_wind_U,
            d_wind_V,
            d_x0,
            d_y0,
            d_release_date,
        ) = (
            cl.Buffer(
                self.context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=hostbuf,
            )
            for hostbuf in (
                self.current_x,
                self.current_y,
                self.current_t,
                self.current_U,
                self.current_V,
                self.wind_x,
                self.wind_y,
                self.wind_t,
                self.wind_U,
                self.wind_V,
                self.x0,
                self.y0,
                self.release_date,
            )
        )
        d_X_out = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.X_out,
        )
        d_Y_out = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.Y_out,
        )
        d_exit_codes = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.exit_code,
        )
        self.buf_time = time.time() - write_start

        # execute the program
        print("\t\tExecuting kernel...")
        execution_start = time.time()
        self.cl_kernel(
            self.queue,
            (len(self.x0),),
            None,
            d_current_x,
            np.uint32(len(self.current_x)),
            d_current_y,
            np.uint32(len(self.current_y)),
            d_current_t,
            np.uint32(len(self.current_t)),
            d_current_U,
            d_current_V,
            d_wind_x,
            np.uint32(len(self.wind_x)),
            d_wind_y,
            np.uint32(len(self.wind_y)),
            d_wind_t,
            np.uint32(len(self.wind_t)),
            d_wind_U,
            d_wind_V,
            d_x0,
            d_y0,
            d_release_date,
            np.uint32(self.advection_scheme),
            np.float64(self.windage_coefficient),
            np.float64(self.eddy_diffusivity),
            self.start_time,
            self.dt,
            self.ntimesteps,
            self.save_every,
            d_X_out,
            d_Y_out,
            d_exit_codes,
        )

        # wait for the computation to complete
        self.queue.finish()
        self.kernel_time = time.time() - execution_start

        # Read back the results from the compute device
        print("\t\tCopying results back to host device...")
        read_start = time.time()
        cl.enqueue_copy(self.queue, self.X_out, d_X_out)
        cl.enqueue_copy(self.queue, self.Y_out, d_Y_out)
        cl.enqueue_copy(self.queue, self.exit_code, d_exit_codes)
        self.buf_time += time.time() - read_start

        # create and return dataset
        P = self.p0.assign_coords({"time": self.out_time})  # add a time dimension
        self.execution_result = P.assign(  # overwrite with new data
            {
                "lon": (["p_id", "time"], self.X_out.reshape([len(P.p_id), -1])),
                "lat": (["p_id", "time"], self.Y_out.reshape([len(P.p_id), -1])),
                "exit_code": (["p_id"], self.exit_code),
            }
        )
        return self.execution_result

    def get_memory_footprint(self) -> dict:
        current_bytes = (
            self.current_x.nbytes
            + self.current_y.nbytes
            + self.current_t.nbytes
            + self.current_U.nbytes
            + self.current_V.nbytes
        )
        wind_bytes = (
            self.wind_x.nbytes
            + self.wind_y.nbytes
            + self.wind_t.nbytes
            + self.wind_U.nbytes
            + self.wind_V.nbytes
        )
        particle_bytes = (
            self.x0.nbytes
            + self.y0.nbytes
            + self.release_date.nbytes
            + self.X_out.nbytes
            + self.Y_out.nbytes
            + self.exit_code.nbytes
        )
        return {
            "current": current_bytes,
            "wind": wind_bytes,
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

        def is_uniformly_spaced_ascending(arr):
            tol = 1e-3
            return len(arr) == 1 or all(np.abs(np.diff(arr) - np.diff(arr)[0]) < tol)

        # check current field valid
        assert max(self.current_x) <= 180
        assert min(self.current_x) >= -180
        assert 1 <= len(self.current_x) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.current_x)
        assert max(self.current_y) <= 90
        assert min(self.current_y) >= -90
        assert 1 <= len(self.current_y) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.current_y)
        assert 1 <= len(self.current_t) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.current_t)

        # check wind field valid
        assert max(self.wind_x) <= 180
        assert min(self.wind_x) >= -180
        assert 1 <= len(self.wind_x) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.wind_x)
        assert max(self.wind_y) <= 90
        assert min(self.wind_y) >= -90
        assert 1 <= len(self.wind_y) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.wind_y)
        assert 1 <= len(self.wind_t) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.wind_t)

        # check particle positions valid
        assert np.nanmax(self.x0) < 180
        assert np.nanmin(self.x0) >= -180
        assert np.nanmax(self.y0) <= 90
        assert np.nanmin(self.y0) >= -90

        # check enum valid
        assert self.advection_scheme in (0, 1)
