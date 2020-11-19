#include "eddy_diffusion.h"

vector eddy_diffusion_meters(double z, const double dt, random_state *state, const double eddy_diffusivity) {
    /*
    Currently, only horizontal diffusion at the surface is supported.
    This diffusion is represented by a random displacement, the magnitude of which is determined
    by the "eddy_diffusivity" parameter.
    */
    vector diff = {.x = 0, .y = 0, .z = 0};
    if (z >= MINIMUM_SURFACE_DIFFUSION_DEPTH) {
        diff.x = (random(state) * 2 - 1) * sqrt(6 * eddy_diffusivity * dt);
        diff.y = (random(state) * 2 - 1) * sqrt(6 * eddy_diffusivity * dt);
    }
    return diff;
}
