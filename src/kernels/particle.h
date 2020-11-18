#ifndef PARTICLE
#define PARTICLE

#include "fields.h"

typedef struct particle {
    int id;
    double x;
    double y;
    double z;
    double t;
} particle;

particle constrain_coordinates(particle p);
particle update_position_no_beaching(particle p, double dx, double dy, double dz, field3d field);
particle update_position(particle p, double dx, double dy, double dz);
void write_p(particle p, __global float *X_out, __global float *Y_out, __global float *Z_out, unsigned int out_timesteps, unsigned int out_idx);
bool is_on_land(particle p, field3d field);
grid_point find_nearest_neighbor(particle p, field3d field);

#endif // PARTICLE
