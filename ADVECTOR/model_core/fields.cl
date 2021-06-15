#include "fields.h"
#include "geography.h"

bool field_element_is_null(field3d field, grid_point gp);

unsigned int find_nearest_neighbor_idx(double value, __global const double *arr, const unsigned int arr_len, const double spacing) {
    /* assumptions:
        -- arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
        -- we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.$
            inside a kernel so we must perform the check in the host code.
        -- value MUST be non-nan.  This function produces UNDEFINED BEHAVIOR if value is nan.
    */
    if (arr == 0 || arr_len <= 1 || isnan(spacing) || spacing == 0) {
        // handles singleton/nonexistent dims and protects against division by zero
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
    if (arr == 0) return 0;  // protect against undefined coordinates

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
    unsigned int z_len = field.z_len ? field.z_len : 1;
    // support for 2d field; if z dimension non-existent, consider it as a singleton
    size_t flat_index = (((gp.t_idx*z_len) + gp.z_idx)*field.y_len + gp.y_idx)*field.x_len + gp.x_idx;
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

double index_bathymetry(field3d field, grid_point gp) {
    /* assumption: gp.[dim]_idx args will be in [0, field.[dim]_len - 1] */
    size_t flat_index = gp.y_idx*field.x_len + gp.x_idx;
    return field.bathy ? field.bathy[flat_index] : NAN;  // ternary expression stops indexing into an undefined variable
}

double calculate_spacing(__global const double *arr, const unsigned int arr_len) {
    if (arr_len > 1) {
        return (arr[arr_len-1] - arr[0]) / (arr_len - 1);
    } else {
        return NAN;
    }
}

bool x_is_circular(field3d field) {
    if (field.x_len < 2) {
        return false;
    }
    double tolerance = .001;
    return (fabs(constrain_longitude_to_valid_domain(field.x[field.x_len - 1] + field.x_spacing) - field.x[0]) < tolerance);
}

bool field_element_is_null(field3d field, grid_point gp) {
    /* returns true if any component of the vector at gp in field is null, and the field defines that component.
      I.e., the element containing vector v can be null even if v.z is nan, if field.W is not defined.*/
    vector v = index_vector_field(field, gp, false);
    return (
        (isnan(v.x) && field.U != 0) ||
        (isnan(v.y) && field.V != 0) ||
        (isnan(v.z) && field.W != 0)
    );
}

vector double_jack_search(grid_point gp, field3d field) {
    /* This function returns some non-null vector from "field" which is nearby grid cell "gp".
        It finds this nearby vector by exploring the x/y dimensions straight outward, and in diagonals, and exploring
          the dimension straight outward.  In 3D this shape is a bit like a Toy Jack, but with two extra axes in the x/y
          dimensions.  Hence, the Double Jack.
        Visual explanation, where the numbers represent the order in which each cell is explored:
        ...                 ...                    ...                 ...
            18     15    19       ^ y                         21               ^ z
                8  5  9           |                           11               |
            12  2  1  3  13          --> x             12  2  1  3  13          --> x
                6  4  7                                       10
            16     14    17                                   20
        ...                 ...                    ...                 ...
        There is no guarantee that the returned vector is the absolute closest vector,
            but this sacrifices accuracy for speed, and is still a decent heuristic.
        If this search explores the whole grid and finds no valid vectors, it returns a vector with NAN components.
    */
    // simplest case: element at gp is non-null
    if (!field_element_is_null(field, gp)) {
        return index_vector_field(field, gp, false);
    }

    // element at gp is null; we must explore the grid!
    // we will explore outwards with radius "r", as far as guarantees we explore the whole grid extent
    unsigned int max_radius = max(
        field.x_is_circular ? field.x_len / 2 : field.x_len,
        max(field.y_len, field.z_len)
    );
    for (unsigned int r = 1; r <= max_radius; r++) {
        // try neighbors in longitude
        for (int sign = -1; sign <= 1; sign += 2) {
            long new_x = gp.x_idx + sign*r;
            if (field.x_is_circular) {
                new_x = (gp.x_idx + sign*r + field.x_len) % field.x_len;
            } else {
                new_x = gp.x_idx + sign*r;
                if ((new_x < 0) || (new_x > field.x_len - 1)) continue;
            }
            grid_point gp_dx = {.x_idx = (unsigned int) new_x, .y_idx = gp.y_idx, .z_idx = gp.z_idx, .t_idx = gp.t_idx};
            if (!field_element_is_null(field, gp_dx)) {
                return index_vector_field(field, gp_dx, false);
            }
        }
        // try neighbors in latitude
        for (int sign = -1; sign <= 1; sign += 2) {
            long new_y = gp.y_idx + sign*r;
            if ((new_y < 0) || (new_y > field.y_len - 1)) continue;
            grid_point gp_dy = {.x_idx = gp.x_idx, .y_idx = (unsigned int) new_y, .z_idx = gp.z_idx, .t_idx = gp.t_idx};
            if (!field_element_is_null(field, gp_dy)) {
                return index_vector_field(field, gp_dy, false);
            }
        }
        // try corners
        for (int lon_sign = -1; lon_sign <= 1; lon_sign += 2) {
            long new_x = gp.x_idx + lon_sign*r;
            if (field.x_is_circular) {
                new_x = (gp.x_idx + lon_sign*r + field.x_len) % field.x_len;
            } else {
                new_x = gp.x_idx + lon_sign*r;
                if ((new_x < 0) || (new_x > field.x_len - 1)) continue;
            }
            for (int lat_sign = -1; lat_sign <= 1; lat_sign += 2) {
                long new_y = gp.y_idx + lat_sign*r;
                if ((new_y < 0) || (new_y > field.y_len - 1)) continue;
                grid_point gp_corner = {.x_idx = (unsigned int) new_x, .y_idx = (unsigned int) new_y,
                                        .z_idx = gp.z_idx, .t_idx = gp.t_idx};
                if (!field_element_is_null(field, gp_corner)) {
                    return index_vector_field(field, gp_corner, false);
                }
            }
        }
        // try neighbors in depth
        for (int sign = -1; sign <= 1; sign += 2) {
            long new_z = gp.z_idx + sign*r;
            if ((new_z < 0) || (new_z > field.z_len - 1)) continue;
            grid_point gp_dz = {.x_idx = gp.x_idx, .y_idx = gp.y_idx, .z_idx = (unsigned int) new_z, .t_idx = gp.t_idx};
            if (!field_element_is_null(field, gp_dz)) {
                return index_vector_field(field, gp_dz, false);
            }
        }
    }
    vector failure = {.x = NAN, .y = NAN, .z = NAN};
    return failure;
}
