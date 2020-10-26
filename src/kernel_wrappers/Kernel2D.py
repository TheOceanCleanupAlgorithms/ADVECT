"""
Since we can't raise errors inside kernels, the best practice is to wrap every kernel in a python object.
Args are passed upon initialization, execution is triggered by method "execute".  Streamlines process
of executing kernels.
"""
import abc
from enum import Enum
from pathlib import Path

import kernels.opencl_specification_constants as cl_const
import numpy as np
import pyopencl as cl
import time

KERNEL_SOURCE = Path(__file__).parent / Path('../kernels/kernel_2d.cl')


class AdvectionScheme(Enum):
    """matching definitions in src/kernels/kernel_2d.cl"""
    eulerian = 0
    taylor2 = 1


class Kernel2D:
    """abstract base class for 2D opencl kernel wrappers"""

    def __init__(self,
                 advection_scheme: AdvectionScheme, eddy_diffusivity: float, context: cl.Context,
                 field_x: np.ndarray, field_y: np.ndarray, field_t: np.ndarray,
                 field_U: np.ndarray, field_V: np.ndarray,
                 x0: np.ndarray, y0: np.ndarray, release_date: np.ndarray,
                 start_time: float, dt: float, ntimesteps: int, save_every: int,
                 X_out: np.ndarray, Y_out: np.ndarray):
        """store args to object, perform argument checking, create opencl objects and some timers"""
        self.field_x, self.field_y, self.field_t = field_x, field_y, field_t
        self.field_U, self.field_V = field_U, field_V
        self.x0, self.y0, self.release_date = x0, y0, release_date
        self.start_time, self.dt, self.ntimesteps, self.save_every = start_time, dt, ntimesteps, save_every
        self.X_out, self.Y_out = X_out, Y_out
        self.advection_scheme = advection_scheme
        self.eddy_diffusivity = eddy_diffusivity
        self._check_args()

        # create opencl objects
        self.context = context
        self.queue = cl.CommandQueue(context)
        self.cl_kernel = cl.Program(context, open(KERNEL_SOURCE).read())\
            .build(options=['-I', str(KERNEL_SOURCE.parent)]).advect

        # some handy timers
        self.buf_time = 0
        self.kernel_time = 0

    def execute(self):
        """tranfers arguments to the compute device, triggers execution, waits on result"""
        # write arguments to compute device
        write_start = time.time()
        d_field_x, d_field_y, d_field_t, d_field_U, d_field_V, d_x0, d_y0, d_release_date = \
            (cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf)
             for hostbuf in
             (self.field_x, self.field_y, self.field_t, self.field_U, self.field_V, self.x0, self.y0, self.release_date))
        d_X_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.X_out)
        d_Y_out = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Y_out)
        self.buf_time = time.time() - write_start

        # execute the program
        self.cl_kernel.set_scalar_arg_dtypes(
                [None, np.uint32, None, np.uint32, None, np.uint32,
                 None, None,
                 None, None, None,
                 np.float64, np.float64, np.uint32, np.uint32,
                 None, None, np.uint32, np.float64])
        execution_start = time.time()
        self.cl_kernel(
                self.queue, (len(self.x0),), None,
                d_field_x, np.uint32(len(self.field_x)),
                d_field_y, np.uint32(len(self.field_y)),
                d_field_t, np.uint32(len(self.field_t)),
                d_field_U, d_field_V,
                d_x0, d_y0, d_release_date,
                np.float64(self.start_time), np.float64(self.dt),
                np.uint32(self.ntimesteps), np.uint32(self.save_every),
                d_X_out, d_Y_out,
                np.uint32(self.advection_scheme.value), np.float64(self.eddy_diffusivity))

        # wait for the computation to complete
        self.queue.finish()
        self.kernel_time = time.time() - execution_start

        # Read back the results from the compute device
        read_start = time.time()
        cl.enqueue_copy(self.queue, self.X_out, d_X_out)
        cl.enqueue_copy(self.queue, self.Y_out, d_Y_out)
        self.buf_time += time.time() - read_start

    def print_memory_footprint(self):
        print('-----MEMORY FOOTPRINT-----')
        coords_bytes = (self.field_x.nbytes + self.field_y.nbytes + self.field_t.nbytes)
        vars_bytes = (self.field_U.nbytes + self.field_V.nbytes)
        particle_bytes = (self.x0.nbytes + self.y0.nbytes + self.release_date.nbytes +
                          self.X_out.nbytes + self.Y_out.nbytes)
        print(f'Field Coordinates:  {coords_bytes / 1e6:10.3f} MB')
        print(f'Field Variables:    {vars_bytes / 1e6:10.3f} MB')
        print(f'Particle Positions: {particle_bytes / 1e6:10.3f} MB')
        print(f'Total:              {(coords_bytes + vars_bytes + particle_bytes) / 1e6:10.3f} MB')
        print('')

    def print_execution_time(self):
        print('------EXECUTION TIME------')
        print(f'Kernel Execution:   {self.kernel_time:10.3f} s')
        print(f'Memory Read/Write:  {self.buf_time:10.3f} s')
        print('')

    def _check_args(self):
        """ensure kernel arguments satisfy constraints"""

        def is_uniformly_spaced(arr):
            tol = 1e-5
            return len(arr) == 1 or all(np.abs(np.diff(arr) - np.diff(arr)[0]) < tol)

        assert max(self.field_x) < 180
        assert min(self.field_x) >= -180
        assert len(self.field_x) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.field_x)

        assert max(self.field_y) < 90
        assert min(self.field_y) >= -90
        assert len(self.field_y) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.field_y)

        assert len(self.field_t) <= cl_const.UINT_MAX + 1
        assert is_uniformly_spaced(self.field_t)

        assert max(self.x0) < 180
        assert min(self.x0) >= -180

        assert max(self.y0) < 90
        assert min(self.y0) >= -90

        assert self.advection_scheme.value in (0, 1)

        assert all(self.release_date >= self.start_time), "you can't release particles before the simulation starts!"
