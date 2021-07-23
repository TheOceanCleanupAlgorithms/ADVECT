import os
from pathlib import Path

import pyopencl as cl

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

ROOT_DIR = Path(__file__).parent.parent
MODEL_CORE_DIR = ROOT_DIR / "ADVECTOR/model_core"
CL_CONTEXT = cl.create_some_context()
CL_QUEUE = cl.CommandQueue(CL_CONTEXT)
