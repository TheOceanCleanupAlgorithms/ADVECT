#ifndef BATHYMETRY
#define BATHYMETRY

#include "vector.h"

typedef struct bathymetry {
    __global const double *x, *y;
    const unsigned int x_len, y_len;
    const double x_spacing, y_spacing;
    __global const float *Z;
} bathymetry;

bool location_is_in_ocean(vector v, bathymetry bathy);

#endif // BATHYMETRY
