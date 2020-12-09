import pyopencl as cl
import os
import numpy as np
from config import ROOT_DIR

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def degrees_lon_to_meters(deg_lon, lat) -> np.ndarray:
    """convert longitude displacement to meters using kernel code
    :param deg_lon: displacement in longitude (degrees E)
    :param lat: latitude of displacement (degrees N)
    """
    # setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, """
    #include "geography.cl"

    __kernel void deg_lon_to_m(
        const double deg_lon,
        const double lat,
        __global double *meters) {

        meters[0] = degrees_lon_to_meters(deg_lon, lat);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    meters = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, meters.nbytes)

    prg.deg_lon_to_m(queue, (1,), None, np.float64(deg_lon), np.float64(lat), d_out)
    queue.finish()

    cl.enqueue_copy(queue, meters, d_out)

    return meters[0]


def degrees_lat_to_meters(deg_lat, lat) -> np.ndarray:
    """convert latitude displacement to meters using kernel code
    :param deg_lat: displacement in latitude (degrees N)
    :param lat: latitude of displacement (degrees N)
    """
    # setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, """
    #include "geography.cl"

    __kernel void deg_lat_to_m(
        const double deg_lat,
        const double lat,
        __global double *meters) {

        meters[0] = degrees_lat_to_meters(deg_lat, lat);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    meters = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, meters.nbytes)

    prg.deg_lat_to_m(queue, (1,), None, np.float64(deg_lat), np.float64(lat), d_out)
    queue.finish()

    cl.enqueue_copy(queue, meters, d_out)

    return meters[0]


# tests go here
