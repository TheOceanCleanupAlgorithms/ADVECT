"""
It seems pyopencl doesn't give you access directly to the constants defined in the opencl specification, e.g. int_max.
So unfortunately they're hard-coded here.  Raises error if the version of OpenCL doesn't match spec we're pulling these
from (OpenCL 2.2)
"""

import pyopencl as cl

assert cl.get_cl_header_version() == (2, 2), "This application requires OpenCL 2.2."

UINT_MAX = 0xFFFFFFFF
