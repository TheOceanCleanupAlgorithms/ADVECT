#ifndef FIELDS
#define FIELDS

#include "vector.h"

typedef struct field2d {
    __global const double *x, *y, *t;
    const unsigned int x_len, y_len, t_len;
    const double x_spacing, y_spacing, t_spacing;
    __global const float *U, *V;
} field2d;

typedef struct grid_point {
    unsigned int x_idx;
    unsigned int y_idx;
    unsigned int t_idx;
} grid_point;

vector index_vector_field(field2d field, grid_point gp, bool zero_nans);
unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing);
double calculate_spacing(__global const double *arr, const unsigned int arr_len);

#endif // FIELDS
