#ifndef FIELDS
#define FIELDS

#include "vector.h"

typedef struct field3d {
    // to represent a 2d field, set .z = 0 and .z_len = 0.  dim_len < 1 is supported for z only.
    __global const double *x, *y, *z, *t;
    const unsigned int x_len, y_len, z_len, t_len;
    const double x_spacing, y_spacing, t_spacing;
    __global const float *U, *V, *W, *bathy;  // set any of these 0 if not present; NANs will be returned from lookups.
    bool x_is_circular;  // explicitly notates whether x is a circular array (e.g. full global dataset vs regional)
} field3d;

typedef struct grid_point {
    unsigned int x_idx;
    unsigned int y_idx;
    unsigned int z_idx;
    unsigned int t_idx;
} grid_point;

vector index_vector_field(field3d field, grid_point gp, bool zero_nans);
double index_bathymetry(field3d field, grid_point gp);
unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing);
unsigned int find_nearest_neighbor_idx_non_uniform(double value, __global const double *arr, const unsigned int arr_len);
double calculate_spacing(__global const double *arr, const unsigned int arr_len);
bool x_is_circular(field3d field);
vector double_jack_search(grid_point gp, field3d field);

#endif // FIELDS
