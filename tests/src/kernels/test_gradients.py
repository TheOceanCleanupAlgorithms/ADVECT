import pyopencl as cl
import numpy as np
from pathlib import Path
from tests.config import ROOT_DIR, CL_CONTEXT, CL_QUEUE

from tests.src.kernels.test_geography import degrees_lon_to_meters, degrees_lat_to_meters

KERNEL_SOURCE = Path(__file__).parent / "test_gradients.cl"
prg = cl.Program(CL_CONTEXT, open(KERNEL_SOURCE).read()).build(
    options=["-I", str(ROOT_DIR / "src/kernels")]
)


def calculate_partials(p: dict, field: dict, x_is_circular: bool = False) -> np.ndarray:
    """calculate partial derivatives of vector field at particle position using kernel code
    :param p: particle location, keys={'x', 'y', 'z', 't'}
    :param field: vector field, keys={'x', 'y', 'z', 't', 'U', 'V', 'W'}.
        x/y/z/t are sorted 1d np arrays; x/y/t uniformly spaced, z ascending
        U/V/W are np ndarrays with shape (t, z, y, x)
    :param x_is_circular: determines how domain is handled.
    :return 4x3 ndarray of partials [[U_x, V_x, W_x],
                                     [U_y, V_y, W_y],
                                     [U_z, V_z, W_z],
                                     [U_t, V_t, W_t]]
    """
    d_field_x, d_field_y, d_field_z, d_field_t, d_field_U, d_field_V, d_field_W = (
        cl.Buffer(
            CL_CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostbuf
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

    partials_out = np.empty(12)
    d_partials_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, partials_out.nbytes)

    prg.test_partials(
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
        np.bool_(x_is_circular),
        d_partials_out,
    )
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, partials_out, d_partials_out)

    return partials_out.reshape([4, 3])


# create some dummy data
default_p = {'x': 0, 'y': 0, 'z': 0, 't': 0}
rng = np.random.default_rng(seed=0)
field = {'x': np.linspace(-2, 2, 10), 'y': np.linspace(-1, 1, 5), 'z': np.linspace(-2, 0, 4), 't': np.linspace(0, 10, 7)}
field['U'] = field['x'] * np.ones((len(field['t']), len(field['z']), len(field['y']), len(field['x'])))
field.update({'V': -1 * field['U'], 'W': 2 * field['U']})


def test_domain_x():
    # test nans returned outside domain, not inside
    p = default_p.copy()
    eps = 1e-7
    domain_min = min(field['x'])
    domain_max = max(field['x'])
    p['x'] = domain_min
    assert all(np.isnan(calculate_partials(p, field)[0]))
    p['x'] = domain_max
    assert all(np.isnan(calculate_partials(p, field)[0]))

    p['x'] = domain_min - eps
    assert all(np.isnan(calculate_partials(p, field)[0]))
    p['x'] = domain_max + eps
    assert all(np.isnan(calculate_partials(p, field)[0]))

    p['x'] = domain_min + eps
    assert not any(np.isnan(calculate_partials(p, field)[0]))
    p['x'] = domain_max - eps
    assert not any(np.isnan(calculate_partials(p, field)[0]))

    # check circular array disables domain checking
    xspacing = (field['x'][-1] - field['x'][0]) / (len(field['x']) - 1)
    p['x'] = domain_min - xspacing * 100
    assert not any(np.isnan(calculate_partials(p, field, x_is_circular=True)[0]))
    p['x'] = domain_max + xspacing * 100
    assert not any(np.isnan(calculate_partials(p, field, x_is_circular=True)[0]))


def test_partial_x():
    xfield = {'x': np.array([0, .5, 1, 1.5, 2, 2.5]), 'y': np.zeros(1), 'z': np.zeros(1), 't': np.zeros(1),
              'U': np.array([0, 2, 7, 9, 5, -1]).reshape((1, 1, 1, 6))}
    xfield.update({'V': -1*xfield['U'], 'W': 2*xfield['U']})
    p = {'x': 0, 'y': 0, 'z': 0, 't': 0}
    dx_m = degrees_lon_to_meters(np.array([.5]), p['y'])[0]

    p['x'] = 0.1  # between index 0 and 1 of xfield.x
    V_xx_true = (2 - 0) / dx_m
    np.testing.assert_allclose(calculate_partials(p, xfield)[0], [V_xx_true, -V_xx_true, 2 * V_xx_true])

    p['x'] = 1.6  # between index 2 and 3 of xfield.x
    V_xx_true = (5 - 9) / dx_m
    np.testing.assert_allclose(calculate_partials(p, xfield)[0], [V_xx_true, -V_xx_true, 2 * V_xx_true])

    p['x'] = -.1  # off bottom of array (but x is circular...)
    V_xx_true = (0 - (-1)) / dx_m
    np.testing.assert_allclose(calculate_partials(p, xfield, x_is_circular=True)[0], [V_xx_true, -V_xx_true, 2 * V_xx_true])
    p['x'] = 2.8  # off top of array (but x is circular...)
    np.testing.assert_allclose(calculate_partials(p, xfield, x_is_circular=True)[0], [V_xx_true, -V_xx_true, 2 * V_xx_true])


def test_domain_y():
    # test nans returned outside domain, not inside
    p = default_p.copy()
    eps = 1e-7

    p['y'] = min(field['y'])
    assert all(np.isnan(calculate_partials(p, field)[1]))
    p['y'] = max(field['y'])
    assert all(np.isnan(calculate_partials(p, field)[1]))

    p['y'] = min(field['y']) - eps
    assert all(np.isnan(calculate_partials(p, field)[1]))
    p['y'] = max(field['y']) + eps
    assert all(np.isnan(calculate_partials(p, field)[1]))

    p['y'] = min(field['y']) + eps
    assert not any(np.isnan(calculate_partials(p, field)[1]))
    p['y'] = max(field['y']) - eps
    assert not any(np.isnan(calculate_partials(p, field)[1]))


def test_partial_y():
    yfield = {'x': np.zeros(1), 'y': np.array([-4, -1, 2, 5, 8]), 'z': np.zeros(1), 't': np.zeros(1),
              'U': np.array([10, -4, 3.6, 7, -3]).reshape((1, 1, 5, 1))}
    yfield.update({'V': -1*yfield['U'], 'W': 2*yfield['U']})
    p = {'x': 0, 'y': 0, 'z': 0, 't': 0}

    p['y'] = 0  # between index 1 and 2 of yfield.y
    V_yx_true = (3.6 - (-4)) / degrees_lat_to_meters(np.array([3]), p['y'])[0]
    np.testing.assert_allclose(calculate_partials(p, yfield)[1], [V_yx_true, -V_yx_true, 2 * V_yx_true])

    p['y'] = 5.0001  # between index 3 and 4 of yfield.y
    V_yx_true = (-3 - 7) / degrees_lat_to_meters(np.array([3]), p['y'])[0]
    np.testing.assert_allclose(calculate_partials(p, yfield)[1], [V_yx_true, -V_yx_true, 2 * V_yx_true])


def test_domain_z():
    # test nans returned outside domain, not inside
    p = default_p.copy()
    eps = 1e-7

    p['z'] = min(field['z'])
    assert all(np.isnan(calculate_partials(p, field)[2]))
    p['z'] = max(field['z'])
    assert all(np.isnan(calculate_partials(p, field)[2]))

    p['z'] = min(field['z']) - eps
    assert all(np.isnan(calculate_partials(p, field)[2]))
    p['z'] = max(field['z']) + eps
    assert all(np.isnan(calculate_partials(p, field)[2]))

    p['z'] = min(field['z']) + eps
    assert not any(np.isnan(calculate_partials(p, field)[2]))
    p['z'] = max(field['z']) - eps
    assert not any(np.isnan(calculate_partials(p, field)[2]))


def test_partial_z():
    zfield = {'x': np.zeros(1), 'y': np.zeros(1), 'z': np.array([-100, -50, -20, -10, -5]), 't': np.zeros(1),
              'U': np.array([4, 52, -2, 4.2, 0]).reshape((1, 5, 1, 1))}
    zfield.update({'V': -1*zfield['U'], 'W': 2*zfield['U']})
    p = {'x': 0, 'y': 0, 'z': 0, 't': 0}

    p['z'] = -80  # between index 0 and 1 of zfield.z
    V_zx_true = (52 - 4) / (-50 - (-100))
    np.testing.assert_allclose(calculate_partials(p, zfield)[2], [V_zx_true, -V_zx_true, 2 * V_zx_true])

    p['z'] = -8  # between index 3 and 4 of zfield.z
    V_zx_true = (0 - 4.2) / (-5 - (-10))
    np.testing.assert_allclose(calculate_partials(p, zfield)[2], [V_zx_true, -V_zx_true, 2 * V_zx_true])


def test_domain_t():
    # test nans returned outside domain, not inside
    p = default_p.copy()
    eps = 1e-7

    p['t'] = min(field['t'])
    assert all(np.isnan(calculate_partials(p, field)[3]))
    p['t'] = max(field['t'])
    assert all(np.isnan(calculate_partials(p, field)[3]))

    p['t'] = min(field['t']) - eps
    assert all(np.isnan(calculate_partials(p, field)[3]))
    p['t'] = max(field['t']) + eps
    assert all(np.isnan(calculate_partials(p, field)[3]))

    p['t'] = min(field['t']) + eps
    assert not any(np.isnan(calculate_partials(p, field)[3]))
    p['t'] = max(field['t']) - eps
    assert not any(np.isnan(calculate_partials(p, field)[3]))


def test_partial_t():
    tfield = {'x': np.zeros(1), 'y': np.zeros(1), 'z': np.zeros(1),
              't': np.array([3410, 3420, 3430, 3440, 3450, 3460]),
              'U': np.array([.5, .8, 10, .2, -.5, 3]).reshape((6, 1, 1, 1))}
    tfield.update({'V': -1*tfield['U'], 'W': 2*tfield['U']})
    p = {'x': 0, 'y': 0, 'z': 0, 't': 0}

    p['t'] = 3422  # between index 1 and 2 of tfield.t
    V_tx_true = (10 - .8) / 10
    np.testing.assert_allclose(calculate_partials(p, tfield)[3], [V_tx_true, -V_tx_true, 2 * V_tx_true])

    p['t'] = 3444  # between index 3 and 4 of tfield.t
    V_tx_true = (-.5 - .2) / 10
    np.testing.assert_allclose(calculate_partials(p, tfield)[3], [V_tx_true, -V_tx_true, 2 * V_tx_true])
