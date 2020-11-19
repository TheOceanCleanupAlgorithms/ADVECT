#ifndef EDDY_DIFFUSION
#define EDDY_DIFFUSION

#include "random.h"
#include "vector.h"

#define MINIMUM_SURFACE_DIFFUSION_DEPTH -1  // depth below which horizontal surface diffusion does not apply

vector eddy_diffusion_meters(double z, const double dt, random_state *state, const double eddy_diffusivity);

#endif // EDDY_DIFFUSION
