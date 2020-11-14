#ifndef WINDAGE
#define WINDAGE

#include "fields.h"
#include "particle.h"

#define MINIMUM_WINDAGE_DEPTH -1  // meters below which windage does not apply

vector windage_meters(particle p, field3d wind, double dt, double windage_coeff);

#endif // WINDAGE
