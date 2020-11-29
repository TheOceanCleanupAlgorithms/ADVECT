#ifndef PARTICLE
#define PARTICLE

#include "fields.h"
#include "bathymetry.h"

typedef struct particle {
    int id;
    double x;
    double y;
    double z;
    double t;
    double r;    // radius
    double rho;  // density
} particle;

particle constrain_coordinates(particle p);
particle update_position_no_beaching(particle p, vector displacement_meters, bathymetry bathy);
particle update_position(particle p, vector displacement_meters);
void write_p(particle p, __global float *X_out, __global float *Y_out, __global float *Z_out, unsigned int out_timesteps, unsigned int out_idx);
bool in_ocean(particle p, bathymetry bathy);
grid_point find_nearest_neighbor(particle p, field3d field);

#endif // PARTICLE
