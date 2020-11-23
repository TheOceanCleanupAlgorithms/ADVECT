#include "fields.h"

unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing) {
    /* assumptions:
        -- arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
        -- we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.$
            inside a kernel so we must perform the check in the host code.
        -- value MUST be non-nan.  This function produces UNDEFINED BEHAVIOR if value is nan.
    */
    if (arr_len == 1 || spacing == 0) {  // handles singletons and protects against division by zero
        return 0;
    } else {
        return (unsigned int) clamp(round((value - arr[0])/spacing), (double) (0.0), (double) (arr_len-1));
    }
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

double calculate_spacing(__global const double *arr, const unsigned int arr_len) {
    if (arr_len > 1) {
        return (arr[arr_len-1] - arr[0]) / (arr_len - 1);
    } else {
        return 0;
    }
}
