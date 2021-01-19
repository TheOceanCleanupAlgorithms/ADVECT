#ifndef EDDY_DIFFUSION
#define EDDY_DIFFUSION

#include "random.h"
#include "vector.h"
#include "vertical_profile.h"


double diffusion_step(double diffusivity, const double dt, random_state *rstate);
vector eddy_diffusion_meters(double z, const double dt, random_state *rstate,
                             vertical_profile horizontal_eddy_diffusivity_profile,
                             vertical_profile vertical_eddy_diffusivity_profile);

#endif // EDDY_DIFFUSION
