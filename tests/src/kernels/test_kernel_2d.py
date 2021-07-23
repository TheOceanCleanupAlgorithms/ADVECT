import pyopencl as cl

from tests.config import CL_CONTEXT, ROOT_DIR, MODEL_CORE_DIR


def test_compiles():
    cl.Program(
        CL_CONTEXT, open(ROOT_DIR / "ADVECTOR/kernels/kernel_2d.cl").read()
    ).build(options=["-I", str(MODEL_CORE_DIR)])
