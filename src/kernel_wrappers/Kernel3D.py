"""
Since we can't raise errors inside kernels, the best practice is to wrap every kernel in a python object.
Args are passed upon initialization, execution is triggered by method "execute".  Streamlines process
of executing kernels.
"""
import numpy as np
import pyopencl as cl
import time
import xarray as xr
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Optional

import kernel_wrappers.kernel_constants as cl_const

KERNEL_SOURCE = Path(__file__).parent / Path('../kernels/kernel_3d.cl')


class AdvectionScheme(Enum):
    """matching definitions in src/kernels/advection_schemes.h"""
    eulerian = 0
    taylor2 = 1


class Kernel3D:
    """wrapper for src/kernels/kernel_3d.cl"""

    def __init__(self, current: xr.Dataset, wind: xr.Dataset, p0: xr.Dataset,
                 advect_time: pd.DatetimeIndex, save_every: int,
                 advection_scheme: AdvectionScheme, eddy_diffusivity: xr.Dataset, windage_multiplier: Optional[float],
                 context: cl.Context):
        """convert convenient python objects to raw representation for kernel"""
        # save some arguments for creating output dataset
        self.p0 = p0
        self.out_time = advect_time[::save_every][1:]

        # ---KERNEL ARGUMENT INITIALIZATION--- #
        # 1-to-1 for arguments in kernel_3d.cl::advect; see comments there for details

        current = current.transpose('time', 'depth', 'lat', 'lon')  # coerce values into correct shape before flattening
        self.current_x = current.lon.values.astype(np.float64)
        self.current_y = current.lat.values.astype(np.float64)
        self.current_z = current.depth.values.astype(np.float64)
        self.current_t = current.time.values.astype('datetime64[s]').astype(np.float64)  # float64 representation of unix timestamp
        self.current_U = current.U.values.astype(np.float32, copy=False).ravel()  # astype will still copy if variable is not already float32
        self.current_V = current.V.values.astype(np.float32, copy=False).ravel()
        self.current_W = current.W.values.astype(np.float32, copy=False).ravel()
        # wind vector field
        if windage_multiplier is not None:
            wind = wind.transpose('time', 'lat', 'lon')  # coerce values into correct shape before flattening
            self.wind_x = wind.lon.values.astype(np.float64)
            self.wind_y = wind.lat.values.astype(np.float64)
            self.wind_t = wind.time.values.astype('datetime64[s]').astype(np.float64)  # float64 representation of unix timestamp
            self.wind_U = wind.U.values.astype(np.float32, copy=False).ravel()  # astype will still copy if variable is not already float32
            self.wind_V = wind.V.values.astype(np.float32, copy=False).ravel()
        else:  # Windage disabled; pass a dummy field with singleton dimensions
            self.wind_x, self.wind_y, self.wind_t = [np.zeros(1, dtype=np.float64)] * 3
            self.wind_U, self.wind_V = [np.zeros((1, 1, 1), dtype=np.float32)] * 2
            self.windage_multiplier = np.nan  # to flag the kernel that windage is disabled
        # particle initialization
        self.x0 = p0.lon.values.astype(np.float32)
        self.y0 = p0.lat.values.astype(np.float32)
        self.z0 = p0.depth.values.astype(np.float32)
        self.release_date = p0.release_date.values.astype('datetime64[s]').astype(np.float64)
        self.radius = p0.radius.values.astype(np.float64)
        self.density = p0.density.values.astype(np.float64)
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
        self.advection_scheme = np.uint32(advection_scheme.value)
        self.windage_multiplier = np.float64(windage_multiplier)
        # eddy diffusivity
        self.horizontal_eddy_diffusivity_z = eddy_diffusivity.z_hd.values.astype(np.float64)
        self.horizontal_eddy_diffusivity_values = eddy_diffusivity.horizontal_diffusivity.values.astype(np.float64)

        # debugging
        self.exit_code = p0.exit_code.values.astype(np.byte)

        # ---HOST INITIALIZATIONS--- #
        # create opencl objects
        self.context = context
        self.queue = cl.CommandQueue(context)
        self.cl_kernel = cl.Program(context, open(KERNEL_SOURCE).read())\
            .build(options=['-I', str(KERNEL_SOURCE.parent)]).advect

        # some handy timers
        self.buf_time = 0
        self.kernel_time = 0

    def execute(self) -> xr.Dataset:
        """tranfers arguments to the compute device, triggers execution, returns result"""
        # perform argument check
        self._check_args()

        # write arguments to compute device
        write_start = time.time()
        d_current_x, d_current_y, d_current_z, d_current_t,\
            d_current_U, d_current_V, d_current_W,\
            d_wind_x, d_wind_y, d_wind_t, d_wind_U, d_wind_V, \
            d_x0, d_y0, d_z0, d_release_date, d_radius, d_density,\
            d_horizontal_eddy_diffusivity_z, d_horizontal_eddy_diffusivity = \
            (cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)
             for hostbuf in
             (self.current_x, self.current_y, self.current_z, self.current_t,
              self.current_U, self.current_V, self.current_W,
              self.wind_x, self.wind_y, self.wind_t, self.wind_U, self.wind_V,
              self.x0, self.y0, self.z0, self.release_date, self.radius, self.density,
              self.horizontal_eddy_diffusivity_z, self.horizontal_eddy_diffusivity_values,
              ))
        d_X_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.X_out)
        d_Y_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Y_out)
        d_Z_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Z_out)
        d_exit_codes = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.exit_code)
        self.buf_time = time.time() - write_start

        # execute the program
        execution_start = time.time()
        self.cl_kernel(
                self.queue, (len(self.x0),), None,
                d_current_x, np.uint32(len(self.current_x)),
                d_current_y, np.uint32(len(self.current_y)),
                d_current_z, np.uint32(len(self.current_z)),
                d_current_t, np.uint32(len(self.current_t)),
                d_current_U, d_current_V, d_current_W,
                d_wind_x, np.uint32(len(self.wind_x)),
                d_wind_y, np.uint32(len(self.wind_y)),
                d_wind_t, np.uint32(len(self.wind_t)),
                d_wind_U, d_wind_V,
                d_x0, d_y0, d_z0, d_release_date, d_radius, d_density,
                self.advection_scheme, self.windage_multiplier,
                d_horizontal_eddy_diffusivity_z, d_horizontal_eddy_diffusivity, np.uint32(len(self.horizontal_eddy_diffusivity_values)),
                self.start_time, self.dt, self.ntimesteps, self.save_every,
                d_X_out, d_Y_out, d_Z_out,
                d_exit_codes)

        # wait for the computation to complete
        self.queue.finish()
        self.kernel_time = time.time() - execution_start

        # Read back the results from the compute device
        read_start = time.time()
        cl.enqueue_copy(self.queue, self.X_out, d_X_out)
        cl.enqueue_copy(self.queue, self.Y_out, d_Y_out)
        cl.enqueue_copy(self.queue, self.Z_out, d_Z_out)
        cl.enqueue_copy(self.queue, self.exit_code, d_exit_codes)
        self.buf_time += time.time() - read_start

        # create and return dataset
        P = self.p0.assign_coords({"time": self.out_time})  # add a time dimension
        return P.assign(  # overwrite with new data
                {
                        "lon": (["p_id", "time"], self.X_out.reshape([len(P.p_id), -1])),
                        "lat": (["p_id", "time"], self.Y_out.reshape([len(P.p_id), -1])),
                        "depth": (["p_id", "time"], self.Z_out.reshape([len(P.p_id), -1])),
                        "exit_code": (["p_id"], self.exit_code),
                }
        )

    def print_memory_footprint(self):
        print('-----MEMORY FOOTPRINT-----')
        current_bytes = (self.current_x.nbytes + self.current_y.nbytes + self.current_z.nbytes + self.current_t.nbytes +
                         self.current_U.nbytes + self.current_V.nbytes + self.current_W.nbytes)
        wind_bytes = (self.wind_x.nbytes + self.wind_y.nbytes + self.wind_t.nbytes +
                      self.wind_U.nbytes + self.wind_V.nbytes)
        particle_bytes = (self.x0.nbytes + self.y0.nbytes + self.release_date.nbytes +
                          self.X_out.nbytes + self.Y_out.nbytes + self.Z_out.nbytes + self.exit_code.nbytes)
        print(f'Current:            {current_bytes / 1e6:10.3f} MB')
        print(f'Wind:               {wind_bytes / 1e6:10.3f} MB')
        print(f'Particle Positions: {particle_bytes / 1e6:10.3f} MB')
        print(f'Total:              {(current_bytes + wind_bytes + particle_bytes) / 1e6:10.3f} MB')
        print('')

    def print_execution_time(self):
        print('------EXECUTION TIME------')
        print(f'Kernel Execution:   {self.kernel_time:10.3f} s')
        print(f'Memory Read/Write:  {self.buf_time:10.3f} s')
        print('')

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

        # check eddy diffusion valid
        assert is_sorted_ascending(self.horizontal_eddy_diffusivity_z)

        # check particle positions valid
        assert np.nanmax(self.x0) < 180
        assert np.nanmin(self.x0) >= -180
        assert np.nanmax(self.y0) <= 90
        assert np.nanmin(self.y0) >= -90
        assert np.nanmax(self.z0) <= 0

        # check enum valid
        assert self.advection_scheme in (0, 1)
