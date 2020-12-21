#include "diffusion.h"

double amplitude_of_diffusion(const double dt, unsigned int ndims, double diffusivity);

vector eddy_diffusion_meters(double z, const double dt, random_state *rstate, vertical_profile kappa_xy_profile) {
    /*
    Currently, only horizontal diffusion is supported.
    This diffusion is represented by a random displacement, the magnitude of which is determined
    by sampling the depth profile of horizontal diffusivity, kappa_xy_profile
    */
    double kappa_xy = sample_profile(kappa_xy_profile, z);
    double horizontal_amplitude = amplitude_of_diffusion(dt, 2, kappa_xy);
    vector diff = {.x = random_within_magnitude(horizontal_amplitude, rstate),
                   .y = random_within_magnitude(horizontal_amplitude, rstate),
                   .z = 0};
    return diff;
}


double amplitude_of_diffusion(const double dt, unsigned int ndims, double diffusivity) {
    /* calculates amplitude of diffusion, given by the square root of the mean square displacement
       (definition of n-dimensional MSD from http://web.mit.edu/savin/Public/.Tutorial_v1.2/Introduction.html)
     * dt: timestep in seconds
     * ndims: number of dimensions of the diffusive process
     * diffusivity: m^2 s^-1
     * return: amplitude in meters
     */
    return sqrt(2 * ndims * diffusivity * dt);
}
