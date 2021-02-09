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

    prg.test_random(CL_QUEUE, (1,), None, np.uint32(seed), np.uint32(num_samples), d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_random():
    nsamples = 100000
    result = random(seed=1, num_samples=nsamples)

    # check range
    assert min(result) > 0
    assert max(result) < 1

    # bin into 10 bins, check each bin has 9-11% of the total samples
    np.testing.assert_allclose(np.histogram(result, bins=10, range=(0, 1))[0] / nsamples, .1, atol=.01)

    # check different seeds produce different values
    seeds = np.arange(1, 10)
    res = [random(seed=s, num_samples=1) for s in seeds]
    assert len(np.unique(res)) == len(seeds)


def random_in_range(low: float, high: float, seed: int, num_samples: int) -> np.ndarray:
    """should return uniform distribution in [-magnitude, magnitude]"""
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "random.cl"

    __kernel void test_random_within_magnitude(
        const double low,
        const double high,
        const unsigned int seed,
        const unsigned int num_samples,
        __global double *out) {

        random_state rstate = {.a = seed};
        for (unsigned int i=0; i<num_samples; i++) {
            out[i] = random_in_range(low, high, &rstate);
        }
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(num_samples).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_random_within_magnitude(
        CL_QUEUE, (1,), None,
        np.float64(low),
        np.float64(high),
        np.uint32(seed),
        np.uint32(num_samples),
        d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_random_in_range():
    nsamples = 100000
    bounds = ((-4, 5), (-13.2, -1.2), (0, 2), (3.2, 5.2))
    for low, high in bounds:
        result = random_in_range(low=low, high=high, seed=1, num_samples=nsamples)

        # check range
        assert min(result) > low
        assert max(result) < high

        # bin into 10 bins, check each bin has 9-11% of the total samples
        np.testing.assert_allclose(np.histogram(result, bins=10, range=(low, high))[0] / nsamples, .1, atol=.01)


def standard_normal(seed: int, num_samples: int) -> np.ndarray:
    """samples from standard normal distribution"""
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "random.cl"

    __kernel void test_standard_normal(
        const unsigned int seed,
        const unsigned int num_samples,
        __global double *out) {

        random_state rstate = {.a = seed};
        for (unsigned int i=0; i<num_samples; i++) {
            out[i] = standard_normal(&rstate);
        }
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(num_samples).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_standard_normal(
        CL_QUEUE, (1,), None,
        np.uint32(seed),
        np.uint32(num_samples),
        d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_standard_normal():
    nsamples = 100000
    result = standard_normal(seed=1, num_samples=nsamples)

    # validate mean and std
    np.testing.assert_allclose(0, np.mean(result), atol=.01)
    np.testing.assert_allclose(1, np.std(result), atol=.01)

    # validate distribution shape
    measured_PDF, bin_edges = np.histogram(result, bins=20, range=(-3, 3), density=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)[0]/2
    true_PDF = 1/np.sqrt(2*np.pi) * np.exp(-.5 * bin_centers**2)
    np.testing.assert_allclose(true_PDF, measured_PDF, atol=.01)


def random_normal(mean: float, std: float, seed: int, num_samples: int) -> np.ndarray:
    """samples from an arbitrary normal distribution"""
    # setup
    prg = cl.Program(CL_CONTEXT, """
    #include "random.cl"

    __kernel void test_random_normal(
        const double mean,
        const double std,
        const unsigned int seed,
        const unsigned int num_samples,
        __global double *out) {

        random_state rstate = {.a = seed};
        for (unsigned int i=0; i<num_samples; i++) {
            out[i] = random_normal(mean, std, &rstate);
        }
    }
    """).build(options=["-I", str(ROOT_DIR / "src/kernels")])

    out = np.zeros(num_samples).astype(np.float64)
    d_out = cl.Buffer(CL_CONTEXT, cl.mem_flags.WRITE_ONLY, out.nbytes)

    prg.test_random_normal(
        CL_QUEUE, (1,), None,
        np.float64(mean),
        np.float64(std),
        np.uint32(seed),
        np.uint32(num_samples),
        d_out)
    CL_QUEUE.finish()

    cl.enqueue_copy(CL_QUEUE, out, d_out)

    return out


def test_random_normal():
    nsamples = 100000
    mean = 23.1
    std = 2.2
    result = random_normal(mean=mean, std=std, seed=1, num_samples=nsamples)

    # validate mean and std
    np.testing.assert_allclose(mean, np.mean(result), atol=.01)
    np.testing.assert_allclose(std, np.std(result), atol=.01)

    # validate distribution shape
    measured_PDF, bin_edges = np.histogram(result, bins=20, range=(mean - 3*std, mean + 3*std), density=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)[0]/2
    true_PDF = 1/(std * np.sqrt(2*np.pi)) * np.exp(-.5 * ((bin_centers - mean)/std)**2)
    np.testing.assert_allclose(true_PDF, measured_PDF, atol=.01)
