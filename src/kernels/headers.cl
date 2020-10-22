#ifndef HEADERS
#define HEADERS
/* houses all the function headers so other files can remain cleaner */
#include "structs.cl"

vector eulerian_displacement(particle p, unsigned int x_idx, unsigned int y_idx, unsigned int t_idx,
                            __global float *field_U, __global float *field_V,
                            __global double *field_x, unsigned int x_len,
                            __global double *field_y, unsigned int y_len,
                            unsigned int t_len, double dt);
vector taylor2_displacement(particle p, unsigned int x_idx, unsigned int y_idx, unsigned int t_idx,
                            __global float *field_U, __global float *field_V,
                            __global double *field_x, unsigned int x_len,
                            __global double *field_y, unsigned int y_len,
                            unsigned int t_len, double dt);
particle constrain_lat_lon(particle p);
particle update_position(particle p, double dx, double dy, double dt);
void write_p(particle p, __global float *X_out, __global float *Y_out, unsigned int out_timesteps, unsigned int out_idx);
unsigned int find_nearest_neighbor_idx(double value, __global double *arr, const unsigned int arr_len, const double spacing);
float index_vector_field(__global float *field, unsigned int x_len, unsigned int y_len,
                         unsigned int x_idx, unsigned int y_idx, unsigned int t_idx);
double degrees_lat_to_meters(double dy, double y);
double degrees_lon_to_meters(double dx, double y);
double meters_to_degrees_lon(double dx_meters, double y);
double meters_to_degrees_lat(double dy_meters, double y);

#endif
