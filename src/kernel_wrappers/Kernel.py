from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyopencl as cl
import xarray as xr

from enums.forcings import Forcing


@dataclass
class KernelConfig(ABC):
    pass


class Kernel(ABC):
    def __init__(
        self,
        forcing_data: Dict[Forcing, xr.Dataset],
        p0: xr.Dataset,
        advect_time: pd.DatetimeIndex,
        save_every: int,
        config: KernelConfig,
        context: cl.Context,
    ):
        """
        :param forcing_data: dict containing forcing datasets
        :param p0: initial state of particles
        :param advect_time: the timeseries which the kernel advects on
        :param save_every: number of timesteps between each writing of particle state
        :param config: dataclass with any extra settings needed by subclass implementation
        :param context: PyopenCL context for executing OpenCL programs
        """
        # save some arguments for creating output dataset
        self.p0 = p0
        self.out_time = advect_time[::save_every][1:]
        self.advect_time = advect_time

        # advection time parameters
        self.start_time = np.float64(advect_time[0].timestamp())
        self.dt = np.float64(pd.Timedelta(advect_time.freq).total_seconds())
        self.ntimesteps = np.uint32(len(advect_time) - 1)  # initial position given
        self.save_every = np.uint32(save_every)

        # create opencl objects
        self.context = context
        self.queue = cl.CommandQueue(context)
        self.cl_kernel = cl.Program(context, self._kernel_source_code).build(
            options=["-I", str(self._model_core_path)]
        )

        # some handy timers
        self.data_load_time = 0
        self.buf_time = 0
        self.kernel_time = 0

    @property
    def _model_core_path(self):
        return Path(__file__).parent.parent / "model_core"

    @property
    @abstractmethod
    def _kernel_source_code(self) -> str:
        pass

    @abstractmethod
    def execute(self) -> xr.Dataset:
        pass

    @abstractmethod
    def get_memory_footprint(self) -> dict:
        pass

    @abstractmethod
    def get_data_loading_time(self) -> float:
        pass

    @abstractmethod
    def get_buffer_transfer_time(self) -> float:
        pass

    @abstractmethod
    def get_kernel_execution_time(self) -> float:
        pass
