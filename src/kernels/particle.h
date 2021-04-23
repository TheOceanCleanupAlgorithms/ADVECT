#ifndef PARTICLE
#define PARTICLE

#include "fields.h"

typedef struct particle {
    int id;
    double x;
    double y;
    double z;
    double t;
    double r;    // radius
    double rho;  // density
    double CSF;  // Corey Shape Factor: c/sqrt(a*b), where a >= b >= c are the dimensions of the particle
} particle;

particle constrain_coordinates(particle p);
particle update_position_no_beaching(particle p, vector displacement_meters, field3d field);
particle update_position(particle p, vector displacement_meters);
void write_p(particle p, __global float *X_out, __global float *Y_out, __global float *Z_out, unsigned int out_timesteps, unsigned int out_idx);
bool in_ocean(particle p, field3d field);
grid_point find_nearest_neighbor(particle p, field3d field);
vector find_nearest_vector(particle p, field3d field, bool zero_nans);
double find_nearest_bathymetry(particle p, field3d field);
vector find_nearby_non_null_vector(particle p, field3d field);

#endif // PARTICLE
