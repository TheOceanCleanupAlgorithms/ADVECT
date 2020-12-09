#include "gradients.h"
#include "geography.h"

/*
    Derivatives are calculated as a finite difference between the two grid points which the particle lies between
        along a given coordinate.  Derivative exactly at gridpoint defined as gradient just above gridpoint (statistically unimportant).
*/

vector x_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt x at p's location
     * units: (m/s) / m
     */
    // note: field.x is a modular (circular) array
    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in x
    grid_point lower = neighbor;  // grid point below particle in x
    if (field.x[neighbor.x_idx] > p.x) {  // particle is below neighbor in x dimension
        if (neighbor.x_idx == 0) {  // off lower end of array; wrap
            lower.x_idx = field.x_len - 1;
        } else {
            lower.x_idx = neighbor.x_idx - 1;
        }
    } else {
        if (neighbor.x_idx == neighbor.x_idx - 1) {  // off upper end of array; wrap
            higher.x_idx = 0;
        } else {
            higher.x_idx = neighbor.x_idx + 1;
        }
    }

    vector V_higher = index_vector_field(field, higher, true);
    vector V_lower = index_vector_field(field, lower, true);

    double dx_m = degrees_lon_to_meters(field.x_spacing, p.y);

    vector V_x;
    V_x.x = (V_higher.x - V_lower.x) / dx_m;
    V_x.y = (V_higher.y - V_lower.y) / dx_m;
    V_x.z = (V_higher.z - V_lower.z) / dx_m;

    return V_x;
}

vector y_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt y at p's location
     * units: (m/s) / m
     */
    if (p.y <= field.y[0] || p.y >= field.y[field.y_len - 1]) {  // position outside of domain
        vector undefined = {.x = NAN, .y = NAN, .z = NAN};
        return undefined;
    }

    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in y
    grid_point lower = neighbor;  // grid point below particle in y
    if (field.y[neighbor.y_idx] > p.y) {  // particle is below neighbor in y dimension
        lower.y_idx = neighbor.y_idx - 1;
    } else {
        higher.y_idx = neighbor.y_idx + 1;
    }

    vector V_higher = index_vector_field(field, higher, true);
    vector V_lower = index_vector_field(field, lower, true);

    double dy_m = degrees_lat_to_meters(field.y_spacing, p.y);

    vector V_y;
    V_y.x = (V_higher.x - V_lower.x) / dy_m;
    V_y.y = (V_higher.y - V_lower.y) / dy_m;
    V_y.z = (V_higher.z - V_lower.z) / dy_m;

    return V_y;
}

vector z_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt z at p's location
     * units: (m/s) / m
     */
    if (p.z <= field.z[0] || p.z >= field.z[field.z_len - 1]) {  // position outside of domain
        vector undefined = {.x = NAN, .y = NAN, .z = NAN};
        return undefined;
    }

    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in z
    grid_point lower = neighbor;  // grid point below particle in z
    if (field.z[neighbor.z_idx] > p.z) {  // particle is below neighbor in z dimension
        lower.z_idx = neighbor.z_idx - 1;
    } else {
        higher.z_idx = neighbor.z_idx + 1;
    }

    vector V_higher = index_vector_field(field, higher, true);
    vector V_lower = index_vector_field(field, lower, true);

    double dz_m = field.z[higher.z_idx] - field.z[lower.z_idx];
    vector V_z;
    V_z.x = (V_higher.x - V_lower.x) / dz_m;
    V_z.y = (V_higher.y - V_lower.y) / dz_m;
    V_z.z = (V_higher.z - V_lower.z) / dz_m;

    return V_z;
}

vector t_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt t at p's location
     * units: (m/s) / s
     */
    if (p.t <= field.t[0] || p.t >= field.t[field.t_len - 1]) {  // position outside of domain
        vector undefined = {.x = NAN, .y = NAN, .z = NAN};
        return undefined;
    }

    grid_point neighbor = find_nearest_neighbor(p, field);
    grid_point higher = neighbor;  // grid point above particle in t
    grid_point lower = neighbor;  // grid point below particle in t
    if (field.t[neighbor.t_idx] > p.t) {  // particle is below neighbor in t dimension
        lower.t_idx = neighbor.t_idx - 1;
    } else {
        higher.t_idx = neighbor.t_idx + 1;
    }

    vector V_higher = index_vector_field(field, higher, true);
    vector V_lower = index_vector_field(field, lower, true);

    vector V_t;
    V_t.x = (V_higher.x - V_lower.x) / field.t_spacing;
    V_t.y = (V_higher.y - V_lower.y) / field.t_spacing;
    V_t.z = (V_higher.z - V_lower.z) / field.t_spacing;

    return V_t;
}