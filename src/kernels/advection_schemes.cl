#include "advection_schemes.h"
#include "geography.h"

vector eulerian_displacement(particle p, field2d field, double dt) {
    // find nearest neighbors in grid
    grid_point neighbor = find_nearest_neighbor(p, field);
    // find U and V nearest to particle position
    vector uv = index_vector_field(field, neighbor, true);

    //////////// advect particle using euler forward advection scheme
    // meters displacement
    vector displacement_meters = {.x = uv.x * dt,
                                  .y = uv.y * dt};
    return displacement_meters;
}

vector taylor2_displacement(particle p, field2d field, double dt) {
    grid_point gp = find_nearest_neighbor(p, field);
    // find adjacent cells
    grid_point gp_w = {.x_idx = (gp.x_idx + 1) % field.x_len, .y_idx = gp.y_idx, .t_idx = gp.t_idx};
    grid_point gp_e = {.x_idx = (gp.x_idx - 1) % field.x_len, .y_idx = gp.y_idx, .t_idx = gp.t_idx};
    grid_point gp_s = {.x_idx = gp.x_idx, .y_idx = max(gp.y_idx - 1, 0u), .t_idx = gp.t_idx};
    grid_point gp_n = {.x_idx = gp.x_idx, .y_idx = min(gp.y_idx + 1, field.y_len - 1), .t_idx = gp.t_idx};
    grid_point gp_dt = {.x_idx = gp.x_idx, .y_idx = gp.y_idx, .t_idx = min(gp.t_idx + 1, field.t_len - 1)};

    // extract values from nearest neighbor + adjacent cells
    vector uv = index_vector_field(field, gp, true);        // at the particle
    vector uv_w = index_vector_field(field, gp_w, true);    // one grid cell left
    vector uv_e = index_vector_field(field, gp_e, true);    // one grid cell right
    vector uv_s = index_vector_field(field, gp_s, true);    // one grid cell down
    vector uv_n = index_vector_field(field, gp_n, true);    // one grid cell up
    vector uv_dt = index_vector_field(field, gp_dt, true);  // one grid cell in future

    double dx = 2 * field.x_spacing; // distance between gp_west and gp_east (deg longitude).
                                     // Uniform spacing, no boundaries, so very simple.
    double dx_m = degrees_lon_to_meters(dx, p.y);


    double dy;  // distance between gp_north and gp_south (deg latitude).
                // Uniform spacing, boundaries at top/bottom, so a bit more complex.
    if (gp.y_idx == 0 || gp.y_idx == field.y_len - 1) {
        dy = field.y_spacing;
    } else {
        dy = 2 * field.y_spacing;
    }
    double dy_m = degrees_lat_to_meters(dy, p.y);

    // Calculate horizontal gradients
    // nan spacing indicates singleton dimension
    double ux = isnan(dx_m) ? 0 : (uv_e.x - uv_w.x) / (dx_m);
    double vx = isnan(dx_m) ? 0 : (uv_e.y - uv_w.y) / (dx_m);
    double uy = isnan(dy_m) ? 0 : (uv_n.x - uv_s.x) / (dy_m);
    double vy = isnan(dy_m) ? 0 : (uv_n.y - uv_s.y) / (dy_m);

    // Calculate time gradients
    double ut = isnan(field.t_spacing) ? 0 : (uv_dt.x - uv.x) / field.t_spacing;
    double vt = isnan(field.t_spacing) ? 0 : (uv_dt.y - uv.y) / field.t_spacing;

    // simplifying term
    double u_ = uv.x + (dt*ut)/2;
    double v_ = uv.y + (dt*vt)/2;

    //////////// advect particle using second-order taylor approx advection scheme (Black and Gay, 1990, eq. 12/13)
    vector displacement_meters;
    displacement_meters.x = (u_ + (uy*v_ - vy*u_) * dt/2) * dt / ((1 - ux*dt/2) * (1 - vy*dt/2) - (uy*vx * pow(dt, 2)) / 4);
    displacement_meters.y = (v_ + (vx*u_ - ux*v_) * dt/2) * dt / ((1 - ux*dt/2) * (1 - vy*dt/2) - (ux*vy * pow(dt, 2)) / 4);

    return displacement_meters;
}
