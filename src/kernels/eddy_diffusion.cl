#include "headers.cl"

double eddy_diffusion_meters(const double dt, random_state *state, const double eddy_diffusivity) {
    /* returns random walk in meters*/
    return (random(state) * 2 - 1) * sqrt(6 * eddy_diffusivity * dt);
}


/* Opencl includes no RNG.  This generator adapted from wikipedia.  Make sure to seed it with
    something which is unique across threads, e.g. the global id.  But also it can't be zero.  So recommended
      would be the global id + 1. */

/* The state must be initialized to non-zero */
double random(random_state *state) {
	/* 32-bit Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	/* calculates random uint, divided by uint_max to produce random double [0, 1]*/
	unsigned int x = state->a;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	state->a = x;
	return x / ((double) UINT_MAX);
}