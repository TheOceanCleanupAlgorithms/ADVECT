#ifndef EDDY_DIFFUSION
#define EDDY_DIFFUSION

#include "random.h"
#include "vector.h"
#include "vertical_profile.h"

#define MINIMUM_SURFACE_DIFFUSION_DEPTH -1  // depth below which horizontal surface diffusion does not apply

vector eddy_diffusion_meters(double z, const double dt, random_state *rstate, vertical_profile kappa_xy_profile);

double diffusion_step(const double dt, unsigned int ndims, double diffusivity, random_state *rstate);

#endif // EDDY_DIFFUSION
