/*
These two physical processes are wrapped together because in the near surface, particle behavior is better defined by
the balance between these two forces than by treating them distinctly.
*/

#ifndef WIND_MIXING_AND_BUOYANCY
#define WIND_MIXING_AND_BUOYANCY

#include "vector.h"
#include "particle.h"
#include "fields.h"
#include "vertical_profile.h"

vector wind_mixing_and_buoyancy_transport(
    particle p, field3d wind, vertical_profile density_profile,
    const double max_wave_height, const double wave_mixing_depth_factor,
    double dt, random_state *rstate, const bool wind_mixing_enabled
);

#endif //WIND_MIXING_AND_BUOYANCY
