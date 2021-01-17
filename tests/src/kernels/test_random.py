import pyopencl as cl
import numpy as np

from tests.config import ROOT_DIR, CL_CONTEXT, CL_QUEUE


def random(seed: int, num_samples: int) -> np.ndarray:
    """should return uniform distribution in [0, 1]"""
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "random.cl"

    __kernel void test_random(
        const unsigned int seed,
        const unsigned int num_samples,
        __global double *out) {

        random_state rstate = {.a = seed};
        for (unsigned int i=0; i<num_samples; i++) {
            out[i] = random(&rstate);
        }
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(num_samples).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_random(CL_QUEUE, (1,), None, np.uint64(seed), np.uint64(num_samples), d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def random_within_magnitude(magnitude: float, seed: int, num_samples: int) -> np.ndarray:
    """should return uniform distribution in [-magnitude, magnitude]"""
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "random.cl"

    __kernel void test_random_within_magnitude(
        const double magnitude,
        const unsigned int seed,
        const unsigned int num_samples,
        __global double *out) {

        random_state rstate = {.a = seed};
        for (unsigned int i=0; i<num_samples; i++) {
            out[i] = random_within_magnitude(magnitude, &rstate);
        }
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(num_samples).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_random_within_magnitude(
        CL_QUEUE, (1,), None, np.float64(magnitude), np.uint64(seed), np.uint64(num_samples), d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_random():
    nsamples = 100000
    result = random(seed=1, num_samples=nsamples)

    # check range
    assert min(result) >= 0
    assert max(result) <= 1

    # bin into 10 bins, check each bin has 9-11% of the total samples
    np.testing.assert_allclose(np.histogram(result, bins=10, range=(0, 1))[0] / nsamples, .1, atol=.01)

    # check different seeds produce different values
    seeds = np.arange(1, 10)
    res = [random(seed=s, num_samples=1) for s in seeds]
    assert len(np.unique(res)) == len(seeds)


def test_random_within_magnitude():
    nsamples = 100000
    magnitude = 50
    result = random_within_magnitude(magnitude=magnitude, seed=1, num_samples=nsamples)

    # check range
    assert min(result) >= -magnitude
    assert max(result) <= magnitude

    # bin into 10 bins, check each bin has 9-11% of the total samples
    np.testing.assert_allclose(np.histogram(result, bins=10, range=(-magnitude, magnitude))[0] / nsamples, .1, atol=.01)


def test_random_in_range():
    assert False, "TODO"
