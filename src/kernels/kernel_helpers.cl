/*
This file contains useful functions for advection tasks, which might be useful to multiple kernels.
*/
typedef struct particle {
    int id;
    float x;
    float y;
    float t;
} particle;

particle constrain_lat_lon(particle p);
particle update_position(particle p, float dx, float dy, float dt);
void write_p(particle p, __global float* X_out, __global float* Y_out, unsigned int out_timesteps, unsigned int out_idx);
unsigned int find_nearest_neighbor_idx(double value, __global double* arr, const unsigned int arr_len, const double spacing);
float index_vector_field(__global float* field, unsigned int x_len, unsigned int y_len,
                         unsigned int x_idx, unsigned int y_idx, unsigned int t_idx);
float degrees_lat_to_meters(float dy, float y);
float degrees_lon_to_meters(float dx, float y);
float meters_to_degrees_lon(float dx_meters, float y);
float meters_to_degrees_lat(float dy_meters, float y);

particle constrain_lat_lon(particle p) {
    // deal with advecting over the poles
    if (p.y > 90) {
        p.y = 180 - p.y;
        p.x = p.x + 180;
    } else if (p.y < -90) {
        p.y = -180 - p.y;
        p.x = p.x + 180;
    }
    // keep longitude representation within [-180, 180)
    // builtin fmod(a, b) is a - b*trunc(a/b), which behaves incorrectly for negative numbers.
    //            so we use  a - b*floor(a/b) instead
    p.x = ((p.x+180) - 360*floor((p.x+180)/360)) - 180;

    return p;
}

particle update_position(particle p, float dx, float dy, float dt) {
    p.x = p.x + dx;
    p.y = p.y + dy;
    p.t = p.t + dt;
    return constrain_lat_lon(p);
}

void write_p(particle p, __global float* X_out, __global float* Y_out, unsigned int out_timesteps, unsigned int out_idx) {
    X_out[p.id*out_timesteps + out_idx] = p.x;
    Y_out[p.id*out_timesteps + out_idx] = p.y;
}

unsigned int find_nearest_neighbor_idx(double value, __global double* arr, const unsigned int arr_len, const double spacing) {
    // assumption: arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
    // also, we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.  Can't raise errors
    // inside a kernel so we must perform the check in the host code.
    return (unsigned int) clamp(round((value - arr[0])/spacing), (double) (0.0), (double) (arr_len-1));
}

float index_vector_field(__global float* field, unsigned int x_len, unsigned int y_len,
                         unsigned int x_idx, unsigned int y_idx, unsigned int t_idx) {
    /*
    field: the vector field from which to retrieve a value.  Dimensions (time, x, y)
    [dim]_len: the length of the [dim] dimension of 'field'
    [dim]_idx: the index along the [dim] dimension
    assumption: [dim]_idx args will be in [0, [dim]_len - 1]
    */
    return field[(t_idx*x_len + x_idx)*y_len + y_idx];
}

// convert meters displacement to lat/lon and back(Reference: American Practical Navigator, Vol II, 1975 Edition, p 5)
float meters_to_degrees_lon(float dx_meters, float y) {
    float rlat = y * M_PI/180;
    return dx_meters / (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

float meters_to_degrees_lat(float dy_meters, float y) {
    float rlat = y * M_PI/180;
    return dy_meters / (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}

float degrees_lon_to_meters(float dx, float y) {
    float rlat = y * M_PI/180;
    return dx * (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

float degrees_lat_to_meters(float dy, float y) {
    float rlat = y * M_PI/180;
    return dy * (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}
