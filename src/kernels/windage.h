#ifndef WINDAGE
#define WINDAGE

#include "fields.h"
#include "particle.h"

vector windage_meters(particle p, field3d wind, double dt, double windage_coeff);

#endif // WINDAGE
