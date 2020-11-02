#include "eddy_diffusion.h"

vector eddy_diffusion_meters(const double dt, random_state *state, const double eddy_diffusivity) {
    /* returns random walk in meters*/
    vector diff = {.x = (random(state) * 2 - 1) * sqrt(6 * eddy_diffusivity * dt),
                   .y = (random(state) * 2 - 1) * sqrt(6 * eddy_diffusivity * dt)};
    return diff;
}
