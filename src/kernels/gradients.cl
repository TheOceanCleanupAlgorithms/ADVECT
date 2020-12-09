#include "gradients.h"

/*
    Derivatives are calculated as a finite difference between the two grid points which the particle lies between
        along a given coordinate.  Derivative exactly at gridpoint defined as gradient just above gridpoint (statistically unimportant).
*/

vector x_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt x at p's location*/
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

    vector V_x;
    V_x.x = (V_higher.x - V_lower.x) / field.x_spacing;
    V_x.y = (V_higher.y - V_lower.y) / field.x_spacing;
    V_x.z = (V_higher.z - V_lower.z) / field.x_spacing;

    return V_x;
}

vector y_partial(particle p, field3d field) {
    /*return the partial derivatives of the vector field wrt x at p's location*/
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

    vector V_y;
    V_y.x = (V_higher.x - V_lower.x) / field.y_spacing;
    V_y.y = (V_higher.y - V_lower.y) / field.y_spacing;
    V_y.z = (V_higher.z - V_lower.z) / field.y_spacing;

    return V_y;
}

vector z_partial(particle p, field3d field) {
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

    double dz = field.z[higher.z_idx] - field.z[lower.z_idx];
    vector V_y;
    V_y.x = (V_higher.x - V_lower.x) / dz;
    V_y.y = (V_higher.y - V_lower.y) / dz;
    V_y.z = (V_higher.z - V_lower.z) / dz;

    return V_y;
}
