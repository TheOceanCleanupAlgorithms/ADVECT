#ifndef WIND_DRIVEN_MIXING
#define WIND_DRIVEN_MIXING

#include "vector.h"
#include "particle.h"
#include "fields.h"
#include "random.h"

vector wind_mixing_meters(particle p, field3d wind, double dt, random_state *rstate);

#endif // WIND_DRIVEN_MIXING
