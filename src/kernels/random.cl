#include "random.h"

double standard_normal(random_state *rstate);

/* Opencl includes no RNG.  This generator adapted from wikipedia.  Make sure to seed it with
    something which is unique across threads, e.g. the global id.  But also it can't be zero.  So recommended
      would be the global id + 1.
   The state must be initialized to non-zero */
double random(random_state *rstate) {
	/* 32-bit Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	/* calculates random uint, divided by uint_max to produce random double [0, 1]*/
	unsigned int x = rstate->a;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rstate->a = x;
	return x / ((double) UINT_MAX);
}

double random_within_magnitude(double magnitude, random_state *rstate) {
    /* returns a uniformly random number in the range (-magnitude, magnitude) */
    return random_in_range(-magnitude, magnitude, rstate);
}

double random_in_range(double low, double high, random_state *rstate) {
    /* returns a uniformly random number in the range (low, high) */
    return low + ((high - low) * random(rstate));
}

double random_normal(double mean, double std, random_state *rstate) {
    /* return: a sample from normal distribution with mean "mean" and standard deviation "std" */
    double standard_normal_sample = standard_normal(rstate);
    return mean + (standard_normal_sample * std);
}

double standard_normal(random_state *rstate) {
    /* returns a sample from the standard normal distribution,
     * generated using the Box-Muller method */
     double a = sqrt(-2*log(random(rstate)));
     double b = 2*M_PI*random(rstate);
     return a * sin(b);
     // notably, since we generated 2 random numbers, it follows that we can get 2 random
     // normal samples.  Indeed, the other one is given by a * cos(b).  We could cache this
     // to halve calls to random, but this would be premature optimization.
}
