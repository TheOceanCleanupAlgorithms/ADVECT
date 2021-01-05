#ifndef VERTICAL_PROFILE
#define VERTICAL_PROFILE

#include "fields.h"

typedef struct vertical_profile {
    __global const double *z;
    const unsigned int len;
    __global const double *value;
} vertical_profile;

double sample_profile(vertical_profile profile, double z);

#endif // VERTICAL_PROFILE
