import pyopencl as cl
import numpy as np

from tests.config import ROOT_DIR, CL_CONTEXT, CL_QUEUE


def degrees_lon_to_meters(deg_lon: np.ndarray, lat: float) -> np.ndarray:
    """convert longitude displacement to meters using kernel code
    :param deg_lon: displacement in longitude (degrees E)
    :param lat: latitude of displacement (degrees N)
    """
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "geography.cl"

    __kernel void deg_lon_to_m(
        __global const double *deg_lon,
        const double lat,
        __global double *meters) {

        meters[get_global_id(0)] = degrees_lon_to_meters(deg_lon[get_global_id(0)], lat);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    d_deg_lon = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=deg_lon.astype(np.float64))
    meters = np.zeros(1).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, meters.nbytes)

    prg.deg_lon_to_m(CL_QUEUE, deg_lon.shape, None, d_deg_lon, np.float64(lat), d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, meters, d_out)

    return meters


def degrees_lat_to_meters(deg_lat: np.ndarray, lat: float) -> np.ndarray:
    """convert latitude displacement to meters using kernel code
    :param deg_lat: displacement in latitude (degrees N)
    :param lat: latitude of displacement (degrees N)
    """
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "geography.cl"

    __kernel void deg_lat_to_m(
        __global const double *deg_lat,
        const double lat,
        __global double *meters) {

        meters[get_global_id(0)] = degrees_lat_to_meters(deg_lat[get_global_id(0)], lat);
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    d_deg_lat = cl.Buffer(CL_CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=deg_lat.astype(np.float64))
    meters = np.zeros_like(deg_lat).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, meters.nbytes)

    prg.deg_lat_to_m(CL_QUEUE, deg_lat.shape, None, d_deg_lat, np.float64(lat), d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, meters, d_out)

    return meters


# tests go here
