#ifndef KERNEL_HELPERS
#define KERNEL_HELPERS
/*
This file contains useful functions for advection tasks, which might be useful to multiple kernels.
*/
#include "structs.cl"

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

particle update_position(particle p, double dx, double dy) {
    p.x = p.x + dx;
    p.y = p.y + dy;
    return constrain_lat_lon(p);
}

void write_p(particle p, __global float *X_out, __global float *Y_out, unsigned int out_timesteps, unsigned int out_idx) {
    X_out[p.id*out_timesteps + out_idx] = (float) p.x;
    Y_out[p.id*out_timesteps + out_idx] = (float) p.y;
}

grid_point find_nearest_neighbor(particle p, field2d field) {
        grid_point neighbor;
        neighbor.x_idx = find_nearest_neighbor_idx(p.x, field.x, field.x_len, field.x_spacing);
        neighbor.y_idx = find_nearest_neighbor_idx(p.y, field.y, field.y_len, field.y_spacing);
        neighbor.t_idx = find_nearest_neighbor_idx(p.t, field.t, field.t_len, field.t_spacing);
        return neighbor;
}

unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing) {
    // assumption: arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
    // also, we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.  Can't raise errors
    // inside a kernel so we must perform the check in the host code.
    return (unsigned int) clamp(round((value - arr[0])/spacing), (double) (0.0), (double) (arr_len-1));
}

vector index_vector_field(field2d field, grid_point gp, bool zero_nans) {
    /*
    assumption: gp.[dim]_idx args will be in [0, field.[dim]_len - 1]
    optional: if zero_nans, any nans encountered will be replaced with zero.  useful for advection schemes.
    */
    vector v = {.x = field.U[(gp.t_idx*field.x_len + gp.x_idx)*field.y_len + gp.y_idx],
                .y = field.V[(gp.t_idx*field.x_len + gp.x_idx)*field.y_len + gp.y_idx]};
    if (zero_nans) {
        if (isnan(v.x)) v.x = 0;
        if (isnan(v.y)) v.y = 0;
    }
    return v;
}

// convert meters displacement to lat/lon and back(Reference: American Practical Navigator, Vol II, 1975 Edition, p 5)
double meters_to_degrees_lon(double dx_meters, double y) {
    double rlat = y * M_PI/180;
    return dx_meters / (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

double meters_to_degrees_lat(double dy_meters, double y) {
    double rlat = y * M_PI/180;
    return dy_meters / (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}

double degrees_lon_to_meters(double dx, double y) {
    double rlat = y * M_PI/180;
    return dx * (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

double degrees_lat_to_meters(double dy, double y) {
    double rlat = y * M_PI/180;
    return dy * (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}

bool is_land(grid_point gp, field2d field) {
    /* where'er you find the vector to be nan,
       you sure as hell can bet that this is land.
        -- William Shakespeare */

    vector nearest_uv = index_vector_field(field, gp, false);
    return (isnan(nearest_uv.x) || isnan(nearest_uv.y));
}

#endif
