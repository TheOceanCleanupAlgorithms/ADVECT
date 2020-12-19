#include "diffusion.h"

vector eddy_diffusion_meters(double z, const double dt, random_state *rstate, const double eddy_diffusivity) {
    /*
    Currently, only horizontal diffusion at the surface is supported.
    This diffusion is represented by a random displacement, the magnitude of which is determined
    by the "eddy_diffusivity" parameter.
    */
    vector diff = {.x = 0, .y = 0, .z = 0};
    if (z >= MINIMUM_SURFACE_DIFFUSION_DEPTH) {
        diff.x = diffusion_step(dt, 2, eddy_diffusivity, rstate);
        diff.y = diffusion_step(dt, 2, eddy_diffusivity, rstate);
    }
    return diff;
}

double diffusion_step(const double dt, unsigned int ndims, double diffusivity, random_state *rstate) {
    /* random step within [-A, A), where A is the amplitude of diffusion, given by the
       square root of the mean square displacement
       (definition of n-dimensional MSD from http://web.mit.edu/savin/Public/.Tutorial_v1.2/Introduction.html)
     * dt: timestep in seconds
     * ndims: number of dimensions
     * diffusivity: m^2 s^-1
     * return: displacement in meters
     */
    return (random(rstate) * 2 - 1) * sqrt(2 * ndims * diffusivity * dt);
}
