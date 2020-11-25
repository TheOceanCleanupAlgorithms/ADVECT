#ifndef DERIVATIVES
#define DERIVATIVES

#include "vector.h"
#include "particle.h"
#include "fields.h"

vector field_derivative_meters(particle p, field2d field, double dt, double derivative_coeff);
vector particle_acceleration_meters(vector previous_speed, vector second_previous_speed, double dt, double acceleration_coeff);

#endif // DERIVATIVES
