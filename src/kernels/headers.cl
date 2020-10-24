#ifndef HEADERS
#define HEADERS
/* houses all the function headers so other files can remain cleaner */
#include "structs.cl"

vector eulerian_displacement(particle p, grid_point neighbor, field2d field, double dt);
vector taylor2_displacement(particle p, grid_point gp, field2d field, double dt);
particle constrain_lat_lon(particle p);
particle update_position(particle p, double dx, double dy);
void write_p(particle p, __global float *X_out, __global float *Y_out, unsigned int out_timesteps, unsigned int out_idx);
grid_point find_nearest_neighbor(particle p, field2d field);
unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing);
vector index_vector_field(field2d field, grid_point gp);
double degrees_lat_to_meters(double dy, double y);
double degrees_lon_to_meters(double dx, double y);
double meters_to_degrees_lon(double dx_meters, double y);
double meters_to_degrees_lat(double dy_meters, double y);
bool is_land(grid_point gp, field2d field);

#endif
