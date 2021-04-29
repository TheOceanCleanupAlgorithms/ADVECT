"""
Since we can't raise errors inside kernels, the best practice is to wrap every kernel in a python object.
Args are passed upon initialization, execution is triggered by method "execute".  Streamlines process
of executing kernels.
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyopencl as cl
import time
import xarray as xr
import pandas as pd
from pathlib import Path

from dask.diagnostics import ProgressBar

import kernel_wrappers.kernel_constants as cl_const
from enums.advection_scheme import AdvectionScheme
from enums.forcings import Forcing
from kernel_wrappers.Kernel import Kernel, KernelConfig

KERNEL_SOURCE = Path(__file__).parent / Path('../kernels/kernel_3d.cl')


@dataclass
class Kernel3DConfig(KernelConfig):
    """ Configuration for 3D Kernel.
    advection_scheme: which mathematical scheme to use for an advection step
    windage_multiplier: scales the physically-motivated windage contribution
    wind_mixing_enabled: toggles wind-driven mixing
    max_wave_height: (m) see config_specifications.md for details
    wave_mixing_depth_factor: see config_specifications.md for details
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
        forcing_data: dict[Forcing, xr.Dataset],
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
        current = forcing_data[Forcing.current].transpose('time', 'depth', 'lat', 'lon')  # coerce values into correct shape before flattening
        with ProgressBar():
            self.current_x = current.lon.values.astype(np.float64)
            self.current_y = current.lat.values.astype(np.float64)
            self.current_z = current.depth.values.astype(np.float64)
            self.current_t = current.time.values.astype('datetime64[s]').astype(np.float64)  # float64 representation of unix timestamp
            self.current_U = current.U.values.astype(np.float32, copy=False).ravel()  # astype will still copy if variable is not already float32
            self.current_V = current.V.values.astype(np.float32, copy=False).ravel()
            self.current_W = current.W.values.astype(np.float32, copy=False).ravel()
            self.current_bathy = current.bathymetry.values.astype(np.float32, copy=False).ravel()
        # wind vector field
        print("\t\tLoading Wind Data...")
        if "wind" in forcing_data:
            with ProgressBar():
                wind = forcing_data[Forcing.wind].transpose('time', 'lat', 'lon')  # coerce values into correct shape before flattening
                self.wind_x = wind.lon.values.astype(np.float64)
                self.wind_y = wind.lat.values.astype(np.float64)
                self.wind_t = wind.time.values.astype('datetime64[s]').astype(np.float64)  # float64 representation of unix timestamp
                self.wind_U = wind.U.values.astype(np.float32, copy=False).ravel()  # astype will still copy if variable is not already float32
                self.wind_V = wind.V.values.astype(np.float32, copy=False).ravel()
        else:  # Windage disabled; pass a dummy field with singleton dimensions
            self.wind_x, self.wind_y, self.wind_t = [np.zeros(1, dtype=np.float64)] * 3
            self.wind_U, self.wind_V = [np.zeros((1, 1, 1), dtype=np.float32)] * 2
            self.windage_multiplier = np.nan  # to flag the kernel that windage is disabled
        # seawater_density vector field
        print("\t\tLoading Seawater Density Data...")
        with ProgressBar():
            seawater_density = forcing_data[Forcing.seawater_density].transpose('time', 'depth', 'lat', 'lon')  # coerce values into correct shape before flattening
            self.seawater_density_x = seawater_density.lon.values.astype(np.float64)
            self.seawater_density_y = seawater_density.lat.values.astype(np.float64)
            self.seawater_density_z = seawater_density.depth.values.astype(np.float64)
            self.seawater_density_t = seawater_density.time.values.astype('datetime64[s]').astype(np.float64)  # float64 representation of unix timestamp
            self.seawater_density_values = seawater_density.rho.values.astype(np.float32, copy=False).ravel()  # astype will still copy if variable is not already float32
        self.data_load_time = time.time() - data_loading_start
        # particle initialization
        self.x0 = p0.lon.values.astype(np.float32)
        self.y0 = p0.lat.values.astype(np.float32)
        self.z0 = p0.depth.values.astype(np.float32)
        self.release_date = p0.release_date.values.astype('datetime64[s]').astype(np.float64)
        self.radius = p0.radius.values.astype(np.float64)
        self.density = p0.density.values.astype(np.float64)
        self.corey_shape_factor = p0.corey_shape_factor.values.astype(np.float64)
        # advection time parameters
        self.start_time = np.float64(advect_time[0].timestamp())
        self.dt = np.float64(pd.Timedelta(advect_time.freq).total_seconds())
        self.ntimesteps = np.uint32(len(advect_time) - 1)  # minus one bc initial position given
        self.save_every = np.uint32(save_every)
        # output_vectors
        self.X_out = np.full((len(p0.lon) * len(self.out_time)), np.nan, dtype=np.float32)  # output will have this value
        self.Y_out = np.full((len(p0.lat) * len(self.out_time)), np.nan, dtype=np.float32)  # until overwritten (e.g. pre-release)
        self.Z_out = np.full((len(p0.depth) * len(self.out_time)), np.nan, dtype=np.float32)
        # physics
        self.advection_scheme = config.advection_scheme.value
        self.windage_multiplier = config.windage_multiplier
        self.wind_mixing_enabled = config.wind_mixing_enabled
        self.max_wave_height = config.max_wave_height
        self.wave_mixing_depth_factor = config.wave_mixing_depth_factor
        # eddy diffusivity
        self.horizontal_eddy_diffusivity_z = config.eddy_diffusivity.z_hd.values.astype(np.float64)
        self.horizontal_eddy_diffusivity_values = config.eddy_diffusivity.horizontal_diffusivity.values.astype(np.float64)
        self.vertical_eddy_diffusivity_z = config.eddy_diffusivity.z_vd.values.astype(np.float64)
        self.vertical_eddy_diffusivity_values = config.eddy_diffusivity.vertical_diffusivity.values.astype(np.float64)
        # debugging
        self.exit_code = p0.exit_code.values.astype(np.byte)

        # ---HOST INITIALIZATIONS--- #
        # create opencl objects
        self.context = context
        self.queue = cl.CommandQueue(context)
        self.cl_kernel = cl.Program(context, open(KERNEL_SOURCE).read())\
            .build(options=['-I', str(KERNEL_SOURCE.parent)]).advect

        self.execution_result = None

    def execute(self) -> xr.Dataset:
        """tranfers arguments to the compute device, triggers execution, returns result"""
        # perform argument check
        self._check_args()

        # write arguments to compute device
        print("\t\tWriting buffers to compute device...")
        write_start = time.time()
        d_current_x, d_current_y, d_current_z, d_current_t,\
            d_current_U, d_current_V, d_current_W, d_current_bathy,\
            d_wind_x, d_wind_y, d_wind_t, d_wind_U, d_wind_V, \
            d_seawater_density_x, d_seawater_density_y, d_seawater_density_z, d_seawater_density_t, \
            d_seawater_density_values, \
            d_x0, d_y0, d_z0, d_release_date, d_radius, d_p_density, d_corey_shape_factor,\
            d_horizontal_eddy_diffusivity_z, d_horizontal_eddy_diffusivity,\
            d_vertical_eddy_diffusivity_z, d_vertical_eddy_diffusivity = \
            (cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)
             for hostbuf in
             (self.current_x, self.current_y, self.current_z, self.current_t,
              self.current_U, self.current_V, self.current_W, self.current_bathy,
              self.wind_x, self.wind_y, self.wind_t, self.wind_U, self.wind_V,
              self.seawater_density_x, self.seawater_density_y, self.seawater_density_z, self.seawater_density_t,
              self.seawater_density_values,
              self.x0, self.y0, self.z0, self.release_date, self.radius, self.density, self.corey_shape_factor,
              self.horizontal_eddy_diffusivity_z, self.horizontal_eddy_diffusivity_values,
              self.vertical_eddy_diffusivity_z, self.vertical_eddy_diffusivity_values,
              ))
        d_X_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.X_out)
        d_Y_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Y_out)
        d_Z_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Z_out)
        d_exit_codes = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.exit_code)
        self.buf_time = time.time() - write_start

        # execute the program
        print("\t\tExecuting kernel...")
        execution_start = time.time()
        self.cl_kernel(
            self.queue, (len(self.x0),), None,
            d_current_x, np.uint32(len(self.current_x)),
            d_current_y, np.uint32(len(self.current_y)),
            d_current_z, np.uint32(len(self.current_z)),
            d_current_t, np.uint32(len(self.current_t)),
            d_current_U, d_current_V, d_current_W, d_current_bathy,
            d_wind_x, np.uint32(len(self.wind_x)),
            d_wind_y, np.uint32(len(self.wind_y)),
            d_wind_t, np.uint32(len(self.wind_t)),
            d_wind_U, d_wind_V,
            d_seawater_density_x, np.uint32(len(self.seawater_density_x)),
            d_seawater_density_y, np.uint32(len(self.seawater_density_y)),
            d_seawater_density_z, np.uint32(len(self.seawater_density_z)),
            d_seawater_density_t, np.uint32(len(self.seawater_density_t)),
            d_seawater_density_values,
            d_x0, d_y0, d_z0, d_release_date, d_radius, d_p_density, d_corey_shape_factor,
            np.uint32(self.advection_scheme), np.float64(self.windage_multiplier), np.uint32(self.wind_mixing_enabled),
            np.float64(self.max_wave_height), np.float64(self.wave_mixing_depth_factor),
            d_horizontal_eddy_diffusivity_z, d_horizontal_eddy_diffusivity, np.uint32(len(self.horizontal_eddy_diffusivity_values)),
            d_vertical_eddy_diffusivity_z, d_vertical_eddy_diffusivity, np.uint32(len(self.vertical_eddy_diffusivity_values)),
            self.start_time, self.dt, self.ntimesteps, self.save_every,
            d_X_out, d_Y_out, d_Z_out,
            d_exit_codes
        )

        # wait for the computation to complete
        self.queue.finish()
        self.kernel_time = time.time() - execution_start

        # Read back the results from the compute device
        print("\t\tCopying results back to host device...")
        read_start = time.time()
        cl.enqueue_copy(self.queue, self.X_out, d_X_out)
        cl.enqueue_copy(self.queue, self.Y_out, d_Y_out)
        cl.enqueue_copy(self.queue, self.Z_out, d_Z_out)
        cl.enqueue_copy(self.queue, self.exit_code, d_exit_codes)
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
        current_bytes = (self.current_x.nbytes + self.current_y.nbytes + self.current_z.nbytes + self.current_t.nbytes +
                         self.current_U.nbytes + self.current_V.nbytes + self.current_W.nbytes)
        wind_bytes = (self.wind_x.nbytes + self.wind_y.nbytes + self.wind_t.nbytes +
                      self.wind_U.nbytes + self.wind_V.nbytes)
        seawater_density_bytes = (
            self.seawater_density_x.nbytes + self.seawater_density_y.nbytes +
            self.seawater_density_z.nbytes + self.seawater_density_t.nbytes +
            self.seawater_density_values.nbytes
        )
        particle_bytes = (self.x0.nbytes + self.y0.nbytes + self.z0.nbytes + self.release_date.nbytes +
                          self.density.nbytes + self.radius.nbytes + self.corey_shape_factor.nbytes +
                          self.X_out.nbytes + self.Y_out.nbytes + self.Z_out.nbytes + self.exit_code.nbytes)
        return {
            "current": current_bytes,
            "wind": wind_bytes,
            "seawater_density": seawater_density_bytes,
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

        def is_sorted_ascending(arr):
            return np.all(np.diff(arr) > 0)

        # check current field valid
        assert max(self.current_x) <= 180
        assert min(self.current_x) >= -180
        assert 1 <= len(self.current_x) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.current_x)
        assert max(self.current_y) <= 90
        assert min(self.current_y) >= -90
        assert 1 <= len(self.current_y) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.current_y)
        assert max(self.current_z) <= 0
        assert is_sorted_ascending(self.current_z)
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

        # check seawater density field valid
        assert max(self.seawater_density_x) <= 180
        assert min(self.seawater_density_x) >= -180
        assert 1 <= len(self.seawater_density_x) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.seawater_density_x)
        assert max(self.seawater_density_y) <= 90
        assert min(self.seawater_density_y) >= -90
        assert 1 <= len(self.seawater_density_y) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced_ascending(self.seawater_density_y)
        assert max(self.seawater_density_z) <= 0
        assert is_sorted_ascending(self.seawater_density_z)
        assert 1 <= len(self.seawater_density_t) <= cl_const.UINT_MAX + 1
        assert is_sorted_ascending(self.seawater_density_t)

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
        assert np.all((.15 < self.corey_shape_factor) & (self.corey_shape_factor <= 1))

        # check enum valid
        assert self.advection_scheme in (0, 1)

        # issue warning if wind timestep is smaller than one day
        if np.any(np.diff(self.wind_t) < pd.Timedelta(days=1).total_seconds()):
            warnings.warn(
                "Timestep of wind data is less than a day.  The kernel assumes a fully developed sea state from each "
                "wind datum; short timesteps mean this is a bad assumption.  Use wind data averaged over a longer "
                "timestep, or complain to the developers (or both)."
            )
