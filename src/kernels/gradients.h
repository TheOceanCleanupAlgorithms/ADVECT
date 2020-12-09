#ifndef GRADIENTS
#define GRADIENTS

#include "vector.h"
#include "fields.h"
#include "particle.h"

vector x_partial(particle p, field3d field);
vector y_partial(particle p, field3d field);
vector z_partial(particle p, field3d field);
vector t_partial(particle p, field3d field);

#endif // GRADIENTS
