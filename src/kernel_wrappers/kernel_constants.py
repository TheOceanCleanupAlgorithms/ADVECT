"""
This file contains constants which are defined within the kernel runtime, hard-copied here in order to make them
accessible to the python runtime.
"""

# this is defined by the OpenCL runtime itself; this matches the definition in the OpenCL 1.2 standard.
UINT_MAX = 0xffffffff

# these match the definitions in kernel_2d.cl.
EXIT_CODES = {0: 'SUCCESS',
              1: 'NULL_LOCATION',
              2: 'INVALID_LATITUDE',
              3: 'PARTICLE_TOO_LARGE',
              -1: 'INVALID_ADVECTION_SCHEME',
              }
