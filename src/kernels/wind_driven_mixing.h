#ifndef WIND_DRIVEN_MIXING
#define WIND_DRIVEN_MIXING

#include "vector.h"
#include "particle.h"
#include "fields.h"
#include "random.h"

#define SURFACE_AIR_DENSITY 1.225  // kg m^-3, according to International Standard Atmosphere (ISO 2533:1975)
#define SURFACE_SEAWATER_DENSITY 1025  // kg m^-3, reasonable appproximation of range at surface (1020-1029)
#define MLD_in_terms_of_wave_height -10  // reasonable approximation, see D'Asaro et al 2013 Figure 1 for evidence
#define WAVE_AGE 25  // Kukulka 2012 assumes 35 (fully developed sea state), but I cannot see where they get this number
    // in their cited source, Komen et al 1996.  Komen does, however, mention 25 as an estimate for an "old wind" sea.
    // as 35 seems to produce too-tall waves, I use 25.

double sample_concentration_profile(double wind_speed_10m, double rise_velocity, random_state *rstate);
double mixed_layer_depth(double wind_speed_10m);

#endif // WIND_DRIVEN_MIXING
