#include "particle.h"

particle constrain_coordinates(particle p) {
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

    // keep particles out of the atmosphere
    if (p.z > 0) {
        p.z = 0;
    }

    return p;
}


particle update_position_no_beaching(particle p, vector displacement_meters, field3d field) {
    /*so-called "slippery coastlines."  Try to move the particle by dx and dy, but avoid depositing it onto land.*/
    particle new_p = update_position(p, displacement_meters);  // always use this to keep lat/lon/depth properly constrained

    // simple case
    if (in_ocean(new_p, field)) return new_p;

    // displacement will move particle out of ocean (aka, onto shoreline, or into bathymetry).
    // We don't want this, so we'll modify displacement (but as little as possible)

    // split displacement into components and sort them
    vector ordered_components[3];
    resolve_and_sort(displacement_meters, ordered_components);

    particle candidates[6];
    // remove a single component, smallest to largest
    candidates[0] = update_position(p, add(ordered_components[1], ordered_components[2]));
    candidates[1] = update_position(p, add(ordered_components[0], ordered_components[2]));
    candidates[2] = update_position(p, add(ordered_components[0], ordered_components[1]));
    // remove two components, smallest to largest
    candidates[3] = update_position(p, ordered_components[2]);
    candidates[4] = update_position(p, ordered_components[1]);
    candidates[5] = update_position(p, ordered_components[0]);

    for (int i=0; i<6; i++) {
        if (in_ocean(candidates[i], field)) return candidates[i];
    }

    // don't move particle at all
    return p;
}

particle update_position(particle p, vector displacement_meters) {
    double dx_deg = meters_to_degrees_lon(displacement_meters.x, p.y);
    double dy_deg = meters_to_degrees_lat(displacement_meters.y, p.y);

    p.x = p.x + dx_deg;
    p.y = p.y + dy_deg;
    p.z = p.z + displacement_meters.z;
    return constrain_coordinates(p);
}

void write_p(particle p, __global float *X_out, __global float *Y_out, __global float *Z_out, unsigned int out_timesteps, unsigned int out_idx) {
    float float_px = (float) p.x;
    // Casting from float to double can in very rare cases transform a value like 179.999993 to 180.0.
    // So in case it happens, we make sure the new value is set to -180.
    if (float_px == 180) 
        float_px = float_px - 360;
    X_out[p.id*out_timesteps + out_idx] = float_px;
    Y_out[p.id*out_timesteps + out_idx] = (float) p.y;
    Z_out[p.id*out_timesteps + out_idx] = (float) p.z;
}

grid_point find_nearest_neighbor(particle p, field3d field) {
/* assumption: particle has non-null latitude and longitude. */
        grid_point neighbor;
        neighbor.x_idx = find_nearest_neighbor_idx(p.x, field.x, field.x_len, field.x_spacing);
        neighbor.y_idx = find_nearest_neighbor_idx(p.y, field.y, field.y_len, field.y_spacing);
        neighbor.z_idx = find_nearest_neighbor_idx_non_uniform(p.z, field.z, field.z_len);
        neighbor.t_idx = find_nearest_neighbor_idx(p.t, field.t, field.t_len, field.t_spacing);
        return neighbor;
}

bool in_ocean(particle p, field3d field) {
    /* where'er you find the vector nan to be,
       you sure as heck can bet this ain't the sea.
        -- William Shakespeare */
    grid_point gp = find_nearest_neighbor(p, field);
    vector V = index_vector_field(field, gp, false);
    bool isnan = (isnan(V.x) || isnan(V.y) || isnan(V.z));
    bool too_deep = p.z < field.z_floor;
    return !(isnan || too_deep);
}
