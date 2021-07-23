from enum import Enum


class AdvectionScheme(Enum):
    """matching definitions in src/kernels/advection_schemes.h"""

    eulerian = 0
    taylor2 = 1
