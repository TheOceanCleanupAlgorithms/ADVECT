#include "advection_schemes.cl"
#include "gradients.cl"
#include "fields.cl"
#include "particle.cl"
#include "geography.cl"

__kernel void test_taylor2(
    /* vector field */
    __global const double *field_x,
    const unsigned int field_x_len,
    __global const double *field_y,
    const unsigned int field_y_len,
    __global const double *field_z,
    const unsigned int field_z_len,
    __global const double *field_t,
    const unsigned int field_t_len,
    __global const float *field_U,
    __global const float *field_V,
    __global const float *field_W,
    /* particle state */
    const double p_x,
    const double p_y,
    const double p_z,
    const double p_t,
    const double dt,
    __global double *displacement_out) {

    field3d field = {.x = field_x, .y = field_y, .z = field_z, .t = field_t,
                     .x_len = field_x_len, .y_len = field_y_len, .z_len = field_z_len, .t_len = field_t_len,
                     .x_spacing = calculate_spacing(field_x, field_x_len),
                     .y_spacing = calculate_spacing(field_y, field_y_len),
                     .t_spacing = calculate_spacing(field_t, field_t_len),
                     .U = field_U, .V = field_V, .W = field_W, .bathy = 0};

    particle p = {.x = p_x, .y = p.y, .z = p_z, .t = p_t};

    vector displacement_meters = taylor2_displacement(p, field, dt);
    displacement_out[0] = displacement_meters.x;
    displacement_out[1] = displacement_meters.y;
    displacement_out[2] = displacement_meters.z;
}
