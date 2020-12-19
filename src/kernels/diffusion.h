#ifndef EDDY_DIFFUSION
#define EDDY_DIFFUSION

#include "random.h"
#include "vector.h"

#define MINIMUM_SURFACE_DIFFUSION_DEPTH -1  // depth below which horizontal surface diffusion does not apply

vector eddy_diffusion_meters(double z, const double dt, random_state *rstate, const double eddy_diffusivity);

double diffusion_step(const double dt, unsigned int ndims, double diffusivity, random_state *rstate);

#endif // EDDY_DIFFUSION
