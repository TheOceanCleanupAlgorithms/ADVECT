#ifndef EDDY_DIFFUSION
#define EDDY_DIFFUSION

#include "random.h"
#include "vector.h"

vector eddy_diffusion_meters(const double dt, random_state *state, const double eddy_diffusivity);

#endif // EDDY_DIFFUSION
