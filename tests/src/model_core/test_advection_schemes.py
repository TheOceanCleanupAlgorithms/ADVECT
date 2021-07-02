from pathlib import Path

import numpy as np
import pyopencl as cl

from tests.config import CL_CONTEXT, CL_QUEUE, MODEL_CORE_DIR

KERNEL_SOURCE = Path(__file__).parent / "test_advection_schemes.cl"


def advect_taylor2(p: dict, field: dict, dt: float) -> np.ndarray:
    """calculate partial derivatives of vector field at particle position using kernel code
    :param p: particle location, keys={'x', 'y', 'z', 't'}
    :param field: vector field, keys={'x', 'y', 'z', 't', 'U', 'V', 'W'}.
        x/y/z/t are sorted 1d np arrays; x/y/t uniformly spaced, z ascending
        U/V/W are np ndarrays with shape (t, z, y, x)
    :param dt: timestep (seconds)
    :return displacement [dx, dy, dz]
    """
    prg = cl.Program(CL_CONTEXT, open(KERNEL_SOURCE).read()).build(
        options=["-I", str(MODEL_CORE_DIR)]
    )
    d_field_x, d_field_y, d_field_z, d_field_t, d_field_U, d_field_V, d_field_W = (
        cl.Buffer(
            CL_CONTEXT,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=hostbuf,
        )
        for hostbuf in (
            field["x"].astype(np.float64),
            field["y"].astype(np.float64),
            field["z"].astype(np.float64),
            field["t"].astype(np.float64),
            field["U"].astype(np.float32).ravel(),
            field["V"].astype(np.float32).ravel(),
            field["W"].astype(np.float32).ravel(),
        )
    )

    displacement_out = np.empty(3)
    d_displacement_out = cl.Buffer(
        CL_CONTEXT, cl.mem_flags.WRITE_ONLY, displacement_out.nbytes
    )

    prg.test_taylor2(
        CL_QUEUE,
        (1,),
        None,
        d_field_x,
        np.uint32(len(field["x"])),
        d_field_y,
        np.uint32(len(field["y"])),
        d_field_z,
        np.uint32(len(field["z"])),
        d_field_t,
        np.uint32(len(field["t"])),
        d_field_U,
        d_field_V,
        d_field_W,
        np.float64(p["x"]),
        np.float64(p["y"]),
        np.float64(p["z"]),
        np.float64(p["t"]),
        np.float64(dt),
        d_displacement_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, displacement_out, d_displacement_out)

    return displacement_out


def taylor2_formula(V, V_x, V_y, V_z, V_t, dt):
    prg = cl.Program(
        CL_CONTEXT,
        """
    #include "advection_schemes.cl"
    __kernel void test_taylor2_formula(
        const double u, const double v, const double w,
        const double ux, const double vx, const double wx,
        const double uy, const double vy, const double wy,
        const double uz, const double vz, const double wz,
        const double ut, const double vt, const double wt,
        const double dt,
        __global double *displacement_out) {
        vector V = {.x = u, .y = v, .z = w};
        vector V_x = {.x = ux, .y = vx, .z = wx};
        vector V_y = {.x = uy, .y = vy, .z = wy};
        vector V_z = {.x = uz, .y = vz, .z = wz};
        vector V_t = {.x = ut, .y = vt, .z = wt};
        
        vector displacement_meters = taylor2_formula(V, V_x, V_y, V_z, V_t, dt);
        displacement_out[0] = displacement_meters.x;
        displacement_out[1] = displacement_meters.y;
        displacement_out[2] = displacement_meters.z;
    } """,
    ).build(options=["-I", str(MODEL_CORE_DIR)])

    displacement_out = np.empty(3)
    d_displacement_out = cl.Buffer(
        CL_CONTEXT, cl.mem_flags.WRITE_ONLY, displacement_out.nbytes
    )

    prg.test_taylor2_formula(
        CL_QUEUE,
        (1,),
        None,
        np.float64(V[0]),
        np.float64(V[1]),
        np.float64(V[2]),
        np.float64(V_x[0]),
        np.float64(V_x[1]),
        np.float64(V_x[2]),
        np.float64(V_y[0]),
        np.float64(V_y[1]),
        np.float64(V_y[2]),
        np.float64(V_z[0]),
        np.float64(V_z[1]),
        np.float64(V_z[2]),
        np.float64(V_t[0]),
        np.float64(V_t[1]),
        np.float64(V_t[2]),
        np.float64(dt),
        d_displacement_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, displacement_out, d_displacement_out)

    return displacement_out


def test_linear_displacement():
    p = {"x": 0, "y": 0, "z": 0, "t": 0}
    field = {
        "x": np.linspace(-2, 2, 10),
        "y": np.linspace(-1, 1, 5),
        "z": np.linspace(-2, 0, 4),
        "t": np.linspace(0, 10, 7),
    }
    field["U"] = np.ones(
        (len(field["t"]), len(field["z"]), len(field["y"]), len(field["x"]))
    )
    field.update({"V": np.zeros_like(field["U"]), "W": np.zeros_like(field["U"])})
    dt = 1
    # test linear displacement in each dimension
    disp = advect_taylor2(p, field, dt)
    np.testing.assert_allclose(disp, [1, 0, 0])

    field.update({"U": field["V"], "V": field["U"]})
    disp = advect_taylor2(p, field, dt)
    np.testing.assert_allclose(disp, [0, 1, 0])

    field.update({"W": field["V"], "V": field["U"]})
    disp = advect_taylor2(p, field, dt)
    np.testing.assert_allclose(disp, [0, 0, 1])

    dt = 2
    disp = advect_taylor2(p, field, dt)
    np.testing.assert_allclose(disp, [0, 0, 2])


def test_dimensional_symmetry():
    rng = np.random.default_rng(seed=0)
    V = rng.random(3)
    V_x = rng.random(3)
    V_y = rng.random(3)
    V_z = rng.random(3)
    V_t = rng.random(3)
    dt = 1

    orig_disp = taylor2_formula(V, V_x, V_y, V_z, V_t, dt)

    # rotate x-->y, y-->z, z-->x
    roll1_disp = taylor2_formula(
        np.roll(V, 1),
        np.roll(V_z, 1),
        np.roll(V_x, 1),
        np.roll(V_y, 1),
        np.roll(V_t, 1),
        dt,
    )

    np.testing.assert_allclose(orig_disp, np.roll(roll1_disp, -1))

    # rotate x-->z, y-->x, z-->y

    roll2_disp = taylor2_formula(
        np.roll(V, 2),
        np.roll(V_y, 2),
        np.roll(V_z, 2),
        np.roll(V_x, 2),
        np.roll(V_t, 2),
        dt,
    )

    np.testing.assert_allclose(orig_disp, np.roll(roll2_disp, -2))
