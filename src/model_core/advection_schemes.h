#ifndef ADVECTION_SCHEMES
#define ADVECTION_SCHEMES

#include "fields.h"
#include "particle.h"
#include "vector.h"

#define EULERIAN 0  // matches definitions in src/enums/advection_scheme.py
#define TAYLOR2 1

vector eulerian_displacement(particle p, field3d field, double dt);
vector taylor2_displacement(particle p, field3d field, double dt);
vector eulerian_displacement_2d(particle p, field3d field, double dt);
vector taylor2_displacement_2d(particle p, field3d field, double dt);

#endif // ADVECTION_SCHEMES
