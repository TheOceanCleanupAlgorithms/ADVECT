import os
import pyopencl as cl
from pathlib import Path
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

ROOT_DIR = Path(__file__).parent.parent

CL_CONTEXT = cl.create_some_context()
CL_QUEUE = cl.CommandQueue(CL_CONTEXT)
