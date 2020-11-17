#include "fields.h"

unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing) {
    // assumption: arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
    // also, we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.  Can't raise errors
    // inside a kernel so we must perform the check in the host code.
    if (arr_len == 1 || spacing == 0) {  // handles singletons and protects against division by zero
        return 0;
    } else {
        return (unsigned int) clamp(round((value - arr[0])/spacing), (double) (0.0), (double) (arr_len-1));
    }
}

unsigned int find_nearest_neighbor_idx_non_uniform(double value, __global const double *arr, const unsigned int arr_len) {
    // we must have arr_len - 1 <= UINT_MAX for index to be representable as an unsigned int.
    // currently a naive search.  if we assume sorted, we can implement a binary search.
    // however, given the nature of the ADVECTOR, fastest is likely to store neighbor grid_point on particle, and do an outward
    // search, since particles don't move much between timesteps, esp. in depth.  Most of the time would be 3 loop iterations.
    unsigned int neighbor_idx = 0;
    double min_distance = INFINITY;
    for (unsigned int i = 0; i < arr_len; i++) {
        double distance = fabs(arr[i] - value);
        if (distance < min_distance) {
            neighbor_idx = i;
            min_distance = distance;
        }
    }
    return neighbor_idx;
}

vector index_vector_field(field3d field, grid_point gp, bool zero_nans) {
    /*
    assumption: gp.[dim]_idx args will be in [0, field.[dim]_len - 1]
    optional: if zero_nans, any nans encountered will be replaced with zero.  useful for advection schemes.
    */
    vector v = {.x = field.U[(((gp.t_idx*field.z_len) + gp.z_idx)*field.x_len + gp.x_idx)*field.y_len + gp.y_idx],
                .y = field.V[(((gp.t_idx*field.z_len) + gp.z_idx)*field.x_len + gp.x_idx)*field.y_len + gp.y_idx]};
    if (zero_nans) {
        if (isnan(v.x)) v.x = 0;
        if (isnan(v.y)) v.y = 0;
    }
    return v;
}

