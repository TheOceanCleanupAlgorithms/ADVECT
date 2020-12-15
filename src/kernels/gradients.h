#ifndef GRADIENTS
#define GRADIENTS

#include "vector.h"
#include "fields.h"
#include "particle.h"

vector x_partial(particle p, field2d field);
vector y_partial(particle p, field2d field);
vector t_partial(particle p, field2d field);

#endif // GRADIENTS
