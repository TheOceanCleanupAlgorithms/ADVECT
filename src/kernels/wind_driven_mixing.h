#ifndef WIND_DRIVEN_MIXING
#define WIND_DRIVEN_MIXING

#include "vector.h"
#include "particle.h"
#include "fields.h"
#include "random.h"

#define MLD_in_terms_of_wave_height -10  // reasonable approximation, see D'Asaro et al 2013 Figure 1 for evidence

double sample_concentration_profile(double wind_speed_10m, double rise_velocity, random_state *rstate);
double mixed_layer_depth(double wind_speed_10m);

#endif // WIND_DRIVEN_MIXING
