import xarray as xr
import pandas as pd
import pyopencl as cl

from abc import ABC, abstractmethod
from enum import Enum


class AdvectionScheme(Enum):
    """matching definitions in src/kernels/advection_schemes.h"""
    eulerian = 0
    taylor2 = 1


class Kernel(ABC):
    @abstractmethod
    def __init__(
            self,
            forcing_data: dict[str, xr.Dataset],
            p0: xr.Dataset,
            advect_time: pd.DatetimeIndex,
            save_every: int,
            advection_scheme: AdvectionScheme,
            config: dict,
            context: cl.Context,
    ):
        """
        :param forcing_data: dict containing forcing datasets; keys in {"current", "wind", "seawater_density"}
        :param p0: initial state of particles
        :param advect_time: the timeseries which the kernel advects on
        :param save_every: number of timesteps between each writing of particle state
        :param advection_scheme: specifies which advection formulation to use
        :param config: dictionary with any extra settings needed by subclass implementation
        :param context: PyopenCL context for executing OpenCL programs
        """
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def get_memory_footprint(self):
        pass
