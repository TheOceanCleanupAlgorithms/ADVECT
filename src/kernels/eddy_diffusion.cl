#include "headers.cl"

#define EDDY_DIFFUSIVITY 1800 // m^2 / s
/* Sylvia Cole et al 2015: diffusivity calculated at a 300km eddy scale, global average in top 1000m, Argo float data.
  This paper shows 2 orders of magnitude variation regionally, not resolving regional differences is a big error source.
  Additionally, the assumption here is that 300km eddies are not resolved by the velocity field itself.  If they are,
  we're doubling up on the eddy transport.  For reference, to resolve 300km eddies, the grid scale probably needs to be
  on order 30km, which at the equator would be ~1/3 degree.
*/

double eddy_diffusion_meters(double dt, random_state *state) {
    /* returns random walk in meters*/
    return (random(state) * 2 - 1) * sqrt(6 * EDDY_DIFFUSIVITY * dt);
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
