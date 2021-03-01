#include "gradients.cl"
#include "fields.cl"
#include "particle.cl"
#include "geography.cl"

__kernel void test_partials(
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
    const unsigned int x_is_circular,
    __global double *partials) {
    
    field3d field = {.x = field_x, .y = field_y, .z = field_z, .t = field_t,
                     .x_len = field_x_len, .y_len = field_y_len, .z_len = field_z_len, .t_len = field_t_len,
                     .x_spacing = calculate_spacing(field_x, field_x_len),
                     .y_spacing = calculate_spacing(field_y, field_y_len),
                     .t_spacing = calculate_spacing(field_t, field_t_len),
                     .U = field_U, .V = field_V, .W = field_W,
                     .z_floor = calculate_coordinate_floor(field_z, field_z_len),  // bottom edge of lowest layer
                     .x_is_circular = x_is_circular};

    particle p = {.x = p_x, .y = p_y, .z = p_z, .t = p_t};

    vector V_x = x_partial(p, field);
    partials[0] = V_x.x;
    partials[1] = V_x.y;
    partials[2] = V_x.z;
    vector V_y = y_partial(p, field);
    partials[3] = V_y.x;
    partials[4] = V_y.y;
    partials[5] = V_y.z;
    vector V_z = z_partial(p, field);
    partials[6] = V_z.x;
    partials[7] = V_z.y;
    partials[8] = V_z.z;
    vector V_t = t_partial(p, field);
    partials[9] = V_t.x;
    partials[10] = V_t.y;
    partials[11] = V_t.z;
}
