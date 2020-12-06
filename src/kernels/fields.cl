#include "fields.h"

unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing) {
    /* assumptions:
        -- arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
        -- we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.$
            inside a kernel so we must perform the check in the host code.
        -- value MUST be non-nan.  This function produces UNDEFINED BEHAVIOR if value is nan.
    */
    if (arr_len == 1 || isnan(spacing) || spacing == 0) {  // handles singletons and protects against division by zero
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
    size_t flat_index = (((gp.t_idx*field.z_len) + gp.z_idx)*field.y_len + gp.y_idx)*field.x_len + gp.x_idx;
    vector V = {.x = field.U ? field.U[flat_index] : NAN, // these ternary expressions serve to stop indexing into
                .y = field.V ? field.V[flat_index] : NAN, // an undefined variable
                .z = field.W ? field.W[flat_index] : NAN};
    if (zero_nans) {
        if (isnan(V.x)) V.x = 0;
        if (isnan(V.y)) V.y = 0;
        if (isnan(V.z)) V.z = 0;
    }
    return V;
}

double calculate_spacing(__global const double *arr, const unsigned int arr_len) {
    if (arr_len > 1) {
        return (arr[arr_len-1] - arr[0]) / (arr_len - 1);
    } else {
        return NAN;
    }
}

double calculate_coordinate_floor(__global const double *arr, const unsigned int arr_len) {
    /*
     * Calculates the lower boundary of a coordinate array, i.e. the edge of the lowest grid cell.
     * arr references the centers of grid cells, hence the extrapolation
    */
    if (arr_len > 1) {
        return arr[0] - (arr[1] - arr[0]) / 2;  // linear extrapolation
    } else if (arr_len == 1) {
        return arr[0];
    } else {
        return NAN;
    }
}
