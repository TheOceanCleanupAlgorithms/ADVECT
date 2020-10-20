from pathlib import Path
from kernel_wrappers.Kernel2D import Kernel2D


class Taylor2Kernel2D(Kernel2D):
    """python wrapper object for function "advect" in taylor2_kernel_2d.cl"""
    def __init__(self, **kwargs):
        """store args to object, perform argument checking, create opencl objects and some timers"""
        super().__init__(Path(__file__).parent / Path('../kernels/taylor2_kernel_2d.cl'), **kwargs)

    def execute(self):
        super().execute()

    def _check_args(self):
        super()._check_args()
