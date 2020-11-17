#include "advection_schemes.h"
#include "geography.h"

vector eulerian_displacement(particle p, field3d field, double dt) {
    // find nearest neighbor in grid
    grid_point neighbor = find_nearest_neighbor(p, field);
    // find vector nearest particle position
    vector V = index_vector_field(field, neighbor, true);

    //////////// advect particle using euler forward advection scheme
    // meters displacement
    vector displacement_meters = {.x = V.x * dt,
                                  .y = V.y * dt,
                                  .z = V.z * dt};
    return displacement_meters;
}

vector taylor2_displacement(particle p, field3d field, double dt) {
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

    // grid spacing at particle in x direction (m)
    double dx;
    if (gp.x_idx == 0) {
        dx = field.x[gp.x_idx + 1] - field.x[gp.x_idx];
    } else {
        dx = field.x[gp.x_idx] - field.x[gp.x_idx - 1];
    }
    double dx_m = degrees_lon_to_meters(dx, p.y);

    // grid spacing at particle in y direction (m)
    double dy;
    if (gp.y_idx == 0) {
        dy = field.y[gp.y_idx + 1] - field.y[gp.y_idx];
    } else {
        dy = field.y[gp.y_idx] - field.y[gp.y_idx - 1];
    }
    double dy_m = degrees_lat_to_meters(dy, p.y);

    // Calculate horizontal gradients
    double ux = (uv_w.x - uv_e.x) / (2*dx_m);
    double vx = (uv_w.y - uv_e.y) / (2*dx_m);
    double uy = (uv_s.x - uv_n.x) / (2*dy_m);
    double vy = (uv_s.y - uv_n.y) / (2*dy_m);

    // Calculate time gradients
    double ut = (uv_dt.x - uv.x) / dt;
    double vt = (uv_dt.y - uv.y) / dt;

    // simplifying term
    double u_ = uv.x + (dt*ut)/2;
    double v_ = uv.y + (dt*vt)/2;

    //////////// advect particle using second-order taylor approx advection scheme (Black and Gay, 1990, eq. 12/13)
    vector displacement_meters;
    displacement_meters.x = (u_ + (uy*v_ - vy*u_) * dt/2) * dt / ((1 - ux*dt/2) * (1 - vy*dt/2) - (uy*vx * pow(dt, 2)) / 4);
    displacement_meters.y = (v_ + (vx*u_ - ux*v_) * dt/2) * dt / ((1 - ux*dt/2) * (1 - vy*dt/2) - (ux*vy * pow(dt, 2)) / 4);

    displacement_meters.z = 0;  // until this gets updated, it does no vertical movement.
    return displacement_meters;
}
