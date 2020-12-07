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
    grid_point gp_west = {.x_idx = (gp.x_idx + 1) % field.x_len, .y_idx = gp.y_idx,
                          .z_idx = gp.z_idx, .t_idx = gp.t_idx};
    grid_point gp_east = {.x_idx = (gp.x_idx - 1) % field.x_len, .y_idx = gp.y_idx,
                          .z_idx = gp.z_idx, .t_idx = gp.t_idx};
    grid_point gp_south = {.x_idx = gp.x_idx, .y_idx = max(gp.y_idx - 1, 0u),
                           .z_idx = gp.z_idx, .t_idx = gp.t_idx};
    grid_point gp_north = {.x_idx = gp.x_idx, .y_idx = min(gp.y_idx + 1, field.y_len - 1),
                           .z_idx = gp.z_idx, .t_idx = gp.t_idx};
    grid_point gp_down = {.x_idx = gp.x_idx, .y_idx = gp.y_idx,
                          .z_idx = max(gp.z_idx - 1, 0u), .t_idx = gp.t_idx};
    grid_point gp_up = {.x_idx = gp.x_idx, .y_idx = gp.y_idx,
                        .z_idx = min(gp.z_idx + 1, field.z_len - 1), .t_idx = gp.t_idx};
    grid_point gp_dt = {.x_idx = gp.x_idx, .y_idx = gp.y_idx,
                        .z_idx = gp.z_idx, .t_idx = min(gp.t_idx + 1, field.t_len - 1)};

    // extract values from nearest neighbor + adjacent cells
    vector V = index_vector_field(field, gp, true);
    vector V_west = index_vector_field(field, gp_west, true);
    vector V_east = index_vector_field(field, gp_east, true);
    vector V_south = index_vector_field(field, gp_south, true);
    vector V_north = index_vector_field(field, gp_north, true);
    vector V_up = index_vector_field(field, gp_up, true);
    vector V_down = index_vector_field(field, gp_down, true);
    vector V_dt = index_vector_field(field, gp_dt, true);


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
    
    // distance between gp_up and gp_down (m).  Non-uniform spacing, boundaries at top/bottom, so most complex.
    double dz_m;
    if (field.z_len == 1) {
        dz_m = NAN;
    } else if (gp.z_idx == 0) { // bottom layer
        dz_m = field.z[gp.z_idx + 1] - field.z[gp.z_idx];
    } else if (gp.z_idx == field.z_len - 1) {  // top layer
        dz_m = field.z[gp.z_idx] - field.z[gp.z_idx - 1];
    } else {
        dz_m = field.z[gp.z_idx + 1] - field.z[gp.z_idx - 1];
    }

    // Calculate horizontal gradients
    // nan spacing means a singleton dimension
    double ux = isnan(dx_m) ? 0 : (V_east.x - V_west.x) / dx_m;
    double vx = isnan(dx_m) ? 0 : (V_east.y - V_west.y) / dx_m;
    double wx = isnan(dx_m) ? 0 : (V_east.z - V_west.z) / dx_m;
    double uy = isnan(dy_m) ? 0 : (V_north.x - V_south.x) / dy_m;
    double vy = isnan(dy_m) ? 0 : (V_north.y - V_south.y) / dy_m;
    double wy = isnan(dy_m) ? 0 : (V_north.z - V_south.z) / dy_m;
    double uz = isnan(dz_m) ? 0 : (V_up.x - V_down.x) / dz_m;
    double vz = isnan(dz_m) ? 0 : (V_up.y - V_down.y) / dz_m;
    double wz = isnan(dz_m) ? 0 : (V_up.z - V_down.z) / dz_m;

    // Calculate time gradients
    double ut = isnan(field.t_spacing) ? 0 : (V_dt.x - V.x) / field.t_spacing;
    double vt = isnan(field.t_spacing) ? 0 : (V_dt.y - V.y) / field.t_spacing;
    double wt = isnan(field.t_spacing) ? 0 : (V_dt.z - V.z) / field.t_spacing;

    // simplifying term
    double u_ = V.x + (dt*ut)/2;
    double v_ = V.y + (dt*vt)/2;
    double w_ = V.z + (dt*wt)/2;

    //////////// advect particle using second-order taylor approx advection scheme
        // simplified form of scheme in Appendix A4 of Tim Jensen Master's Thesis; copied from his code
        // at https://github.com/TimJansen94/3D-dispersal-model-Tim/blob/master/run_model/funcs_advect.m
    vector displacement_meters;
    displacement_meters.x = dt * -(8*u_ - 4*dt*u_*vy + 4*dt*uy*v_ - 4*dt*u_*wz + 4*dt*uz*w_ + 2*pow(dt, 2)*u_*vy*wz - 2*pow(dt, 2)*u_*vz*wy - 2*pow(dt, 2)*uy*v_*wz + 2*pow(dt, 2)*uy*vz*w_ + 2*pow(dt, 2)*uz*v_*wy - 2*pow(dt, 2)*uz*vy*w_)/(4*dt*ux + 4*dt*vy + 4*dt*wz - 2*pow(dt, 2)*ux*vy + 2*pow(dt, 2)*uy*vx - 2*pow(dt, 2)*ux*wz + 2*pow(dt, 2)*uz*wx - 2*pow(dt, 2)*vy*wz + 2*pow(dt, 2)*vz*wy + pow(dt, 3)*ux*vy*wz - pow(dt, 3)*ux*vz*wy - pow(dt, 3)*uy*vx*wz + pow(dt, 3)*uy*vz*wx + pow(dt, 3)*uz*vx*wy - pow(dt, 3)*uz*vy*wx - 8);
    displacement_meters.y = dt * -(8*v_ + 4*dt*u_*vx - 4*dt*ux*v_ - 4*dt*v_*wz + 4*dt*vz*w_ - 2*pow(dt, 2)*u_*vx*wz + 2*pow(dt, 2)*u_*vz*wx + 2*pow(dt, 2)*ux*v_*wz - 2*pow(dt, 2)*ux*vz*w_ - 2*pow(dt, 2)*uz*v_*wx + 2*pow(dt, 2)*uz*vx*w_)/(4*dt*ux + 4*dt*vy + 4*dt*wz - 2*pow(dt, 2)*ux*vy + 2*pow(dt, 2)*uy*vx - 2*pow(dt, 2)*ux*wz + 2*pow(dt, 2)*uz*wx - 2*pow(dt, 2)*vy*wz + 2*pow(dt, 2)*vz*wy + pow(dt, 3)*ux*vy*wz - pow(dt, 3)*ux*vz*wy - pow(dt, 3)*uy*vx*wz + pow(dt, 3)*uy*vz*wx + pow(dt, 3)*uz*vx*wy - pow(dt, 3)*uz*vy*wx - 8);
    displacement_meters.z = dt * -(8*w_ + 4*dt*u_*wx - 4*dt*ux*w_ + 4*dt*v_*wy - 4*dt*vy*w_ + 2*pow(dt, 2)*u_*vx*wy - 2*pow(dt, 2)*u_*vy*wx - 2*pow(dt, 2)*ux*v_*wy + 2*pow(dt, 2)*ux*vy*w_ + 2*pow(dt, 2)*uy*v_*wx - 2*pow(dt, 2)*uy*vx*w_)/(4*dt*ux + 4*dt*vy + 4*dt*wz - 2*pow(dt, 2)*ux*vy + 2*pow(dt, 2)*uy*vx - 2*pow(dt, 2)*ux*wz + 2*pow(dt, 2)*uz*wx - 2*pow(dt, 2)*vy*wz + 2*pow(dt, 2)*vz*wy + pow(dt, 3)*ux*vy*wz - pow(dt, 3)*ux*vz*wy - pow(dt, 3)*uy*vx*wz + pow(dt, 3)*uy*vz*wx + pow(dt, 3)*uz*vx*wy - pow(dt, 3)*uz*vy*wx - 8);
            
    return displacement_meters;
}
