#ifndef EDDY_DIFFUSION
#define EDDY_DIFFUSION

#include "random.h"
#include "vector.h"
#include "vertical_profile.h"

vector eddy_diffusion_meters(double z, const double dt, random_state *rstate, vertical_profile kappa_xy_profile);

#endif // EDDY_DIFFUSION
