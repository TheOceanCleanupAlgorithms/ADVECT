#include "particle.h"

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


particle update_position_no_beaching(particle p, double dx, double dy, field3d field) {
    /*so-called "slippery coastlines."  Try to move the particle by dx and dy, but avoid depositing it onto land.*/
    particle new_p = update_position(p, dx, dy);  // always use this to keep lat/lon properly constrained

    // simple case
    if (!is_on_land(new_p, field)) return new_p;

    // particle will beach.  We don't want this, but we do want to try to move the particle in at least one direction.
    particle p_dx = update_position(p, dx, 0);  // only move in x direction
    particle p_dy = update_position(p, 0, dy);  // only move in y direction
    bool is_sea_x = !is_on_land(p_dx, field);
    bool is_sea_y = !is_on_land(p_dy, field);

    if (is_sea_x && is_sea_y) {  // could move in x OR y.  This is like being at a peninsula.
        if (degrees_lon_to_meters(dx, p.y) > degrees_lat_to_meters(dy, p.y)) {
            return p_dx;            // we choose which way to go based on which vector component is stronger.
        } else {
            return p_dy;
        }
    } else if (is_sea_x) {      // we can only move in x; this is like being against a horizontal coastline
        return p_dx;
    } else if (is_sea_y) {      // we can only move in y; this is like being against a vertical coastline
        return p_dy;
    } else {                    // we can't move in x or y; this is like being in a corner, surrounded by land
        return p;
    }
}

particle update_position(particle p, double dx, double dy) {
    p.x = p.x + dx;
    p.y = p.y + dy;
    return constrain_lat_lon(p);
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

bool is_on_land(particle p, field3d field) {
    /* where'er you find the vector to be nan,
       you sure as hell can bet that this is land.
        -- William Shakespeare */
    grid_point gp = find_nearest_neighbor(p, field);
    vector nearest_uv = index_vector_field(field, gp, false);
    return (isnan(nearest_uv.x) || isnan(nearest_uv.y));
}
