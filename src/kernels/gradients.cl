#include "gradients.h"
#include "geography.h"

/*
    Derivatives are calculated as a finite difference between the two grid points which the particle lies between
        along a given coordinate.  Derivative exactly at gridpoint defined as gradient just above gridpoint (statistically unimportant).
    Derivatives undefined at and beyond grid bounds, except in the special case that x is a circular array
        (e.g. global longitude), in which case p beyond grid bounds is assumed within domain,
        with the derivative defined in a circular manner.
*/

__constant vector UNDEFINED = {.x = NAN, .y = NAN, .z = NAN};

vector calculate_partial(grid_point lower, grid_point higher, double spacing, field3d field);
bool in_domain(double position, __global const double *dim, const unsigned int dim_len);

vector calculate_partial(grid_point lower, grid_point higher, double spacing, field3d field) {
    vector V_higher = index_vector_field(field, higher, true);
    vector V_lower = index_vector_field(field, lower, true);

    vector partial;
    partial.x = (V_higher.x - V_lower.x) / spacing;
    partial.y = (V_higher.y - V_lower.y) / spacing;
    partial.z = (V_higher.z - V_lower.z) / spacing;
    return partial;
}

bool in_domain(double position, __global const double *dim, const unsigned int dim_len) {
    return (position > dim[0] && position < dim[dim_len - 1]);
}

vector x_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt x at p's location
     * units: (m/s) / m
     */
    // note: field.x is considered a modular (circular) array

    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in x
    grid_point lower = neighbor;  // grid point below particle in x
    // if x is circular, assumed if p is outside domain, it is "between" points.
    if (!in_domain(p.x, field.x, field.x_len)) {
        if (!field.x_is_circular) { // p outside valid domain
            return UNDEFINED;
        } else {  // p is assumed above last, below first (bc x circular)
            lower.x_idx = field.x_len - 1;
            higher.x_idx = 0;
        }
    } else {
        if (field.x[neighbor.x_idx] > p.x) {  // particle is below neighbor in x dimension
            lower.x_idx = neighbor.x_idx - 1;
        } else {
            higher.x_idx = neighbor.x_idx + 1;
        }
    }

    double dx_m = degrees_lon_to_meters(field.x_spacing, p.y);
    return calculate_partial(lower, higher, dx_m, field);
}

vector y_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt y at p's location
     * units: (m/s) / m
     */
    if (!in_domain(p.y, field.y, field.y_len)) {  // position outside of domain
        return UNDEFINED;
    }

    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in y
    grid_point lower = neighbor;  // grid point below particle in y
    if (field.y[neighbor.y_idx] > p.y) {  // particle is below neighbor in y dimension
        lower.y_idx = neighbor.y_idx - 1;
    } else {
        higher.y_idx = neighbor.y_idx + 1;
    }

    double dy_m = degrees_lat_to_meters(field.y_spacing, p.y);
    return calculate_partial(lower, higher, dy_m, field);
}

vector z_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt z at p's location
     * units: (m/s) / m
     */
    if (!in_domain(p.z, field.z, field.z_len)) {  // position outside of domain
        return UNDEFINED;
    }

    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in z
    grid_point lower = neighbor;  // grid point below particle in z
    if (field.z[neighbor.z_idx] > p.z) {  // particle is below neighbor in z dimension
        lower.z_idx = neighbor.z_idx - 1;
    } else {
        higher.z_idx = neighbor.z_idx + 1;
    }

    double dz_m = field.z[higher.z_idx] - field.z[lower.z_idx];
    return calculate_partial(lower, higher, dz_m, field);
}

vector t_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt t at p's location
     * units: (m/s) / s
     */
    if (!in_domain(p.t, field.t, field.t_len)) {  // position outside of domain
        return UNDEFINED;
    }

    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in t
    grid_point lower = neighbor;  // grid point below particle in t
    if (field.t[neighbor.t_idx] > p.t) {  // particle is below neighbor in t dimension
        lower.t_idx = neighbor.t_idx - 1;
    } else {
        higher.t_idx = neighbor.t_idx + 1;
    }

    return calculate_partial(lower, higher, field.t_spacing, field);
}
