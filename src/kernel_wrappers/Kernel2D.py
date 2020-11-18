"""
Since we can't raise errors inside kernels, the best practice is to wrap every kernel in a python object.
Args are passed upon initialization, execution is triggered by method "execute".  Streamlines process
of executing kernels.
"""
from enum import Enum
from pathlib import Path
from typing import Optional

import kernel_wrappers.kernel_constants as cl_const
import numpy as np
import pyopencl as cl
import time

KERNEL_SOURCE = Path(__file__).parent / Path('../kernels/kernel_2d.cl')


class AdvectionScheme(Enum):
    """matching definitions in src/kernels/kernel_2d.cl"""
    eulerian = 0
    taylor2 = 1


class Kernel2D:
    """wrapper for src/kernels/kernel_2d.cl"""

    def __init__(self,
                 context: cl.Context,
                 current_x: np.ndarray, current_y: np.ndarray, current_t: np.ndarray,
                 current_U: np.ndarray, current_V: np.ndarray,
                 wind_x: np.ndarray, wind_y: np.ndarray, wind_t: np.ndarray,
                 wind_U: np.ndarray, wind_V: np.ndarray,
                 x0: np.ndarray, y0: np.ndarray, release_date: np.ndarray,
                 start_time: float, dt: float, ntimesteps: int, save_every: int,
                 advection_scheme: AdvectionScheme, eddy_diffusivity: float, windage_coeff: Optional[float],
                 X_out: np.ndarray, Y_out: np.ndarray, exit_code: np.ndarray):
        """store args to object, perform argument checking, create opencl objects and some timers"""
        self.current_x, self.current_y, self.current_t = current_x, current_y, current_t
        self.current_U, self.current_V = current_U, current_V
        if windage_coeff is not None:
            self.wind_x, self.wind_y, self.wind_t = wind_x, wind_y, wind_t
            self.wind_U, self.wind_V = wind_U, wind_V
        else:  # opencl won't pass totally empty arrays to the kernel.  Windage disabled, so array contents don't matter
            self.wind_x, self.wind_y, self.wind_t = [np.zeros(1, dtype=np.float64)] * 3
            self.wind_U, self.wind_V = [np.zeros((1, 1, 1), dtype=np.float32)] * 2
            self.windage_coeff = np.nan  # to flag the kernel that windage is disabled
        self.x0, self.y0, self.release_date = x0, y0, release_date
        self.start_time, self.dt, self.ntimesteps, self.save_every = start_time, dt, ntimesteps, save_every
        self.X_out, self.Y_out = X_out, Y_out
        self.advection_scheme = advection_scheme
        self.eddy_diffusivity = eddy_diffusivity
        self.windage_coeff = windage_coeff
        self._check_args()

        # create opencl objects
        self.context = context
        self.queue = cl.CommandQueue(context)
        self.cl_kernel = cl.Program(context, open(KERNEL_SOURCE).read())\
            .build(options=['-I', str(KERNEL_SOURCE.parent)]).advect

        # some handy timers
        self.buf_time = 0
        self.kernel_time = 0

        # debugging
        self.exit_code = exit_code

    def execute(self):
        """tranfers arguments to the compute device, triggers execution, waits on result"""
        # write arguments to compute device
        write_start = time.time()
        d_current_x, d_current_y, d_current_t, d_current_U, d_current_V, \
            d_wind_x, d_wind_y, d_wind_t, d_wind_U, d_wind_V, \
                d_x0, d_y0, d_release_date = \
            (cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)
             for hostbuf in
             (self.current_x, self.current_y, self.current_t, self.current_U, self.current_V,
              self.wind_x, self.wind_y, self.wind_t, self.wind_U, self.wind_V,
              self.x0, self.y0, self.release_date))
        d_X_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.X_out)
        d_Y_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Y_out)
        d_exit_codes = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.exit_code)
        self.buf_time = time.time() - write_start

        # execute the program
        self.cl_kernel.set_scalar_arg_dtypes(
                [None, np.uint32, None, np.uint32, None, np.uint32,
                 None, None,
                 None, np.uint32, None, np.uint32, None, np.uint32,
                 None, None,
                 None, None, None,
                 np.float64, np.float64, np.uint32, np.uint32,
                 None, None,
                 np.uint32, np.float64, np.float64,
                 None])
        execution_start = time.time()
        self.cl_kernel(
                self.queue, (len(self.x0),), None,
                d_current_x, np.uint32(len(self.current_x)),
                d_current_y, np.uint32(len(self.current_y)),
                d_current_t, np.uint32(len(self.current_t)),
                d_current_U, d_current_V,
                d_wind_x, np.uint32(len(self.wind_x)),
                d_wind_y, np.uint32(len(self.wind_y)),
                d_wind_t, np.uint32(len(self.wind_t)),
                d_wind_U, d_wind_V,
                d_x0, d_y0, d_release_date,
                np.float64(self.start_time), np.float64(self.dt),
                np.uint32(self.ntimesteps), np.uint32(self.save_every),
                d_X_out, d_Y_out,
                np.uint32(self.advection_scheme.value), np.float64(self.eddy_diffusivity), np.float64(self.windage_coeff),
                d_exit_codes)

        # wait for the computation to complete
        self.queue.finish()
        self.kernel_time = time.time() - execution_start

        # Read back the results from the compute device
        read_start = time.time()
        cl.enqueue_copy(self.queue, self.X_out, d_X_out)
        cl.enqueue_copy(self.queue, self.Y_out, d_Y_out)
        cl.enqueue_copy(self.queue, self.exit_code, d_exit_codes)
        self.buf_time += time.time() - read_start

    def print_memory_footprint(self):
        print('-----MEMORY FOOTPRINT-----')
        current_bytes = (self.current_x.nbytes + self.current_y.nbytes + self.current_t.nbytes +
                         self.current_U.nbytes + self.current_V.nbytes)
        wind_bytes = (self.wind_x.nbytes + self.wind_y.nbytes + self.wind_t.nbytes +
                      self.wind_U.nbytes + self.wind_V.nbytes)
        particle_bytes = (self.x0.nbytes + self.y0.nbytes + self.release_date.nbytes +
                          self.X_out.nbytes + self.Y_out.nbytes + self.exit_code.nbytes)
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

        def is_uniformly_spaced(arr):
            tol = 1e-3
            return len(arr) == 1 or all(np.abs(np.diff(arr) - np.diff(arr)[0]) < tol)

        # check current field valid
        assert max(self.current_x) <= 180
        assert min(self.current_x) >= -180
        assert 1 <= len(self.current_x) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.current_x)
        assert max(self.current_y) <= 90
        assert min(self.current_y) >= -90
        assert 1 <= len(self.current_y) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.current_y)
        assert 1 <= len(self.current_t) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.current_t)

        # check wind field valid
        assert max(self.wind_x) <= 180
        assert min(self.wind_x) >= -180
        assert 1 <= len(self.wind_x) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.wind_x)
        assert max(self.wind_y) <= 90
        assert min(self.wind_y) >= -90
        assert 1 <= len(self.wind_y) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.wind_y)
        assert 1 <= len(self.wind_t) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.wind_t)

        # check particle positions valid
        assert np.nanmax(self.x0) < 180
        assert np.nanmin(self.x0) >= -180
        assert np.nanmax(self.y0) <= 90
        assert np.nanmin(self.y0) >= -90

        # check enum valid
        assert self.advection_scheme.value in (0, 1)
