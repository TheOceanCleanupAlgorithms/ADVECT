#ifndef ADVECTION_SCHEMES
#define ADVECTION_SCHEMES

#include "fields.h"
#include "particle.h"
#include "vector.h"

#define EULERIAN 0  // matches definitions in src/kernel_wrappers/Kernel2D.py
#define TAYLOR2 1

vector eulerian_displacement(particle p, grid_point neighbor, field2d field, double dt);
vector taylor2_displacement(particle p, grid_point gp, field2d field, double dt);

#endif // ADVECTION_SCHEMES
