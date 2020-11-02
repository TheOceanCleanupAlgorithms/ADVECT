#ifndef WINDAGE
#define WINDAGE

#include "fields.h"
#include "particle.h"

vector windage_meters(particle p, field2d wind, double dt, double windage_coeff);

#endif // WINDAGE
