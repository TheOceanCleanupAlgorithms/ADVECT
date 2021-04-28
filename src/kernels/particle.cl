#include "particle.h"
#include "geography.h"

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
    p.x = constrain_longitude_to_valid_domain(p.x);

    // keep particles out of the atmosphere
    if (p.z > 0) {
        p.z = 0;
    }

    return p;
}


particle update_position_no_beaching_3d(particle p, vector displacement_meters, field3d field) {
    /*
     So-called "slippery coastlines," expanded to 3d.
     Try to move the particle by dx and dy, but avoid depositing it onto land.
    */
    particle new_p = update_position(p, displacement_meters);  // always use this to keep lat/lon/depth properly constrained

    // simple case
    if (in_ocean(new_p, field)) return new_p;

    // displacement will move particle out of ocean (aka, onto shoreline, or into bathymetry).
    // We don't want this, so we'll modify displacement (but as little as possible)

    // first, we modify the displacement's z component based on bathymetry
    // this is essential to allow huge vertical displacements to clip to the bathymetry.
    double bathy_at_p = find_nearest_bathymetry(p, field);
    double bathy_at_new_p = find_nearest_bathymetry(new_p, field);
    // we should try and push it up as little as possible; fmin determines the clipping point.
    // then the fmax clips the z up if it's below the clipping point.
    double clipped_z = fmax(new_p.z, fmin(bathy_at_p, bathy_at_new_p));
    displacement_meters.z = clipped_z - p.z;

    // check simple case again now that displacement has been adjusted to match bathymetry
    new_p = update_position(p, displacement_meters);
    if (in_ocean(new_p, field)) return new_p;

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


particle update_position_no_beaching_2d(particle p, vector displacement_meters, field3d field) {
    /*so-called "slippery coastlines."  Try to move the particle by dx and dy, but avoid depositing it onto land.*/
    particle new_p = update_position(p, displacement_meters);  // always use this to keep lat/lon properly constrained

    // simple case
    if (in_ocean_2d(new_p, field)) return new_p;

    // particle will beach.  We don't want this, but we do want to try to move the particle in at least one direction.
    vector only_x = {.x = displacement_meters.x};
    particle p_dx = update_position(p, only_x);  // only move in x direction
    vector only_y = {.y = displacement_meters.y};
    particle p_dy = update_position(p, only_y);  // only move in y direction
    bool is_sea_x = in_ocean_2d(p_dx, field);
    bool is_sea_y = in_ocean_2d(p_dy, field);

    if (is_sea_x && is_sea_y) {  // could move in x OR y.  This is like being at a peninsula.
        if (degrees_lon_to_meters(displacement_meters.x, p.y) > degrees_lat_to_meters(displacement_meters.y, p.y)) {
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

particle update_position(particle p, vector displacement_meters) {
    double dx_deg = meters_to_degrees_lon(displacement_meters.x, p.y);
    double dy_deg = meters_to_degrees_lat(displacement_meters.y, p.y);

    p.x = p.x + dx_deg;
    p.y = p.y + dy_deg;
    p.z = p.z + displacement_meters.z;
    return constrain_coordinates(p);
}

void write_p_2d(particle p, __global float *X_out, __global float *Y_out, unsigned int out_timesteps, unsigned int out_idx) {
    float float_px = (float) p.x;
    // Casting from float to double can in very rare cases transform a value like 179.999993 to 180.0.
    // So in case it happens, we make sure the new value is set to -180.
    if (float_px == 180) 
        float_px = float_px - 360;
    X_out[p.id*out_timesteps + out_idx] = float_px;
    Y_out[p.id*out_timesteps + out_idx] = (float) p.y;
}

void write_p(particle p, __global float *X_out, __global float *Y_out, __global float *Z_out, unsigned int out_timesteps, unsigned int out_idx) {
    write_p_2d(p, X_out, Y_out, out_timesteps, out_idx);
    Z_out[p.id*out_timesteps + out_idx] = (float) p.z;
}

grid_point find_nearest_neighbor(particle p, field3d field) {
/* assumption: particle has non-null latitude and longitude. */
    grid_point neighbor;
    neighbor.x_idx = find_nearest_neighbor_idx(p.x, field.x, field.x_len, field.x_spacing);
    neighbor.y_idx = find_nearest_neighbor_idx(p.y, field.y, field.y_len, field.y_spacing);
    neighbor.z_idx = find_nearest_neighbor_idx_non_uniform(p.z, field.z, field.z_len);
    neighbor.t_idx = isnan(field.t_spacing) ?
        find_nearest_neighbor_idx_non_uniform(p.t, field.t, field.t_len) :
        find_nearest_neighbor_idx(p.t, field.t, field.t_len, field.t_spacing);
    return neighbor;
}

vector find_nearest_vector(particle p, field3d field, bool zero_nans) {
    return index_vector_field(field, find_nearest_neighbor(p, field), zero_nans);
}

double find_nearest_bathymetry(particle p, field3d field) {
    grid_point neighbor = {
        .x_idx = find_nearest_neighbor_idx(p.x, field.x, field.x_len, field.x_spacing),
        .y_idx = find_nearest_neighbor_idx(p.y, field.y, field.y_len, field.y_spacing),
    };
    return index_bathymetry(field, neighbor);
}

vector find_nearby_non_null_vector(particle p, field3d field) {
    return double_jack_search(find_nearest_neighbor(p, field), field);
}

bool in_ocean(particle p, field3d field) {
    /* where'er you are below bathymetry,
       you sure as heck can bet this ain't the sea.
        -- William Shakespeare */
    return p.z >= find_nearest_bathymetry(p, field);
}

bool in_ocean_2d(particle p, field3d field) {
    /* where'er you find the vector nan to be,
       you sure as heck can bet this ain't the sea.
        -- William Shakespeare */
    grid_point gp = find_nearest_neighbor(p, field);
    vector nearest_uv = index_vector_field(field, gp, false);
    return (!isnan(nearest_uv.x) && !isnan(nearest_uv.y));
}
