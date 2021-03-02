#include "fields.cl"

__kernel void test_field_element_is_null(
    /* vector field */
    __global const double *x, const unsigned int x_len,
    __global const double *y, const unsigned int y_len,
    __global const double *z, const unsigned int z_len,
    __global const double *t, const unsigned int t_len,
    __global const float *U, const unsigned int U_enabled,
    __global const float *V, const unsigned int V_enabled,
    __global const float *W, const unsigned int W_enabled,
    /* grid point */
    const unsigned int x_idx,
    const unsigned int y_idx,
    const unsigned int z_idx,
    const unsigned int t_idx,
    __global unsigned int *out) {
    field3d field = {
        .x = x, .x_len = x_len, .x_spacing = calculate_spacing(x, x_len),
        .y = y, .y_len = y_len, .y_spacing = calculate_spacing(y, y_len),
        .z = z, .z_len = z_len,
        .t = t, .t_len = t_len, .t_spacing = calculate_spacing(t, t_len),
        .U = U_enabled ? U : 0,
        .V = V_enabled ? V : 0,
        .W = W_enabled ? W : 0,
    };
    grid_point gp = {
        .x_idx = x_idx,
        .y_idx = y_idx,
        .z_idx = z_idx,
        .t_idx = t_idx,
    };
    out[0] = field_element_is_null(field, gp);
}


__kernel void test_double_cross_search(
    /* vector field */
    __global const double *x, const unsigned int x_len,
    __global const double *y, const unsigned int y_len,
    __global const double *z, const unsigned int z_len,
    __global const double *t, const unsigned int t_len,
    __global const float *U,
    const unsigned int x_is_circular,  // boolean
    /* grid point */
    const unsigned int x_idx,
    const unsigned int y_idx,
    const unsigned int z_idx,
    const unsigned int t_idx,
    __global double *out) {
    field3d field = {
        .x = x, .x_len = x_len, .x_spacing = calculate_spacing(x, x_len),
        .y = y, .y_len = y_len, .y_spacing = calculate_spacing(y, y_len),
        .z = z, .z_len = z_len,
        .t = t, .t_len = t_len, .t_spacing = calculate_spacing(t, t_len),
        .U = U, .V = 0, .W = 0,
        .x_is_circular = (bool) x_is_circular,
    };
    grid_point gp = {
        .x_idx = x_idx,
        .y_idx = y_idx,
        .z_idx = z_idx,
        .t_idx = t_idx,
    };
    vector nearby = double_cross_search(gp, field);
    out[0] = nearby.x;
    out[1] = nearby.y;
    out[2] = nearby.z;
}
