#include "advection_schemes.h"
#include "gradients.h"

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

    // extract values from nearest neighbor
    vector V = index_vector_field(field, gp, true);

    vector V_x = x_partial(p, field);
    vector V_y = y_partial(p, field);
    vector V_z = z_partial(p, field);
    vector V_t = t_partial(p, field);

    // Calculate horizontal gradients
    // nan spacing means a singleton dimension
    double ux = isnan(V_x.x) ? 0 : V_x.x;
    double vx = isnan(V_x.y) ? 0 : V_x.y;
    double wx = isnan(V_x.z) ? 0 : V_x.z;
    double uy = isnan(V_y.x) ? 0 : V_y.x;
    double vy = isnan(V_y.y) ? 0 : V_y.y;
    double wy = isnan(V_y.z) ? 0 : V_y.z;
    double uz = isnan(V_z.x) ? 0 : V_z.x;
    double vz = isnan(V_z.y) ? 0 : V_z.y;
    double wz = isnan(V_z.z) ? 0 : V_z.z;
    double ut = isnan(V_t.x) ? 0 : V_t.x;
    double vt = isnan(V_t.y) ? 0 : V_t.y;
    double wt = isnan(V_t.z) ? 0 : V_t.z;

    // simplifying term
    double u_ = V.x + (dt*ut)/2;
    double v_ = V.y + (dt*vt)/2;
    double w_ = V.z + (dt*wt)/2;

    //////////// advect particle using second-order taylor approx advection scheme
        // simplified form of scheme in Appendix A4 of Tim Jensen Master's Thesis; copied from his code
        // at https://github.com/TimJansen94/3D-dispersal-model-Tim/blob/master/run_model/funcs_advect.m
    vector displacement_meters;
    displacement_meters.x = dt * -(8*u_ - 4*dt*u_*vy + 4*dt*uy*v_ - 4*dt*u_*wz + 4*dt*uz*w_ + 2*pow(dt, 2)*u_*vy*wz
                                   - 2*pow(dt, 2)*u_*vz*wy - 2*pow(dt, 2)*uy*v_*wz + 2*pow(dt, 2)*uy*vz*w_
                                   + 2*pow(dt, 2)*uz*v_*wy - 2*pow(dt, 2)*uz*vy*w_)
                                / (4*dt*ux + 4*dt*vy + 4*dt*wz - 2*pow(dt, 2)*ux*vy + 2*pow(dt, 2)*uy*vx
                                   - 2*pow(dt, 2)*ux*wz + 2*pow(dt, 2)*uz*wx - 2*pow(dt, 2)*vy*wz + 2*pow(dt, 2)*vz*wy
                                   + pow(dt, 3)*ux*vy*wz - pow(dt, 3)*ux*vz*wy - pow(dt, 3)*uy*vx*wz
                                   + pow(dt, 3)*uy*vz*wx + pow(dt, 3)*uz*vx*wy - pow(dt, 3)*uz*vy*wx - 8);
    displacement_meters.y = dt * -(8*v_ + 4*dt*u_*vx - 4*dt*ux*v_ - 4*dt*v_*wz + 4*dt*vz*w_ - 2*pow(dt, 2)*u_*vx*wz
                                   + 2*pow(dt, 2)*u_*vz*wx + 2*pow(dt, 2)*ux*v_*wz - 2*pow(dt, 2)*ux*vz*w_
                                   - 2*pow(dt, 2)*uz*v_*wx + 2*pow(dt, 2)*uz*vx*w_)
                                / (4*dt*ux + 4*dt*vy + 4*dt*wz - 2*pow(dt, 2)*ux*vy + 2*pow(dt, 2)*uy*vx
                                   - 2*pow(dt, 2)*ux*wz + 2*pow(dt, 2)*uz*wx - 2*pow(dt, 2)*vy*wz + 2*pow(dt, 2)*vz*wy
                                   + pow(dt, 3)*ux*vy*wz - pow(dt, 3)*ux*vz*wy - pow(dt, 3)*uy*vx*wz
                                   + pow(dt, 3)*uy*vz*wx + pow(dt, 3)*uz*vx*wy - pow(dt, 3)*uz*vy*wx - 8);
    displacement_meters.z = dt * -(8*w_ + 4*dt*u_*wx - 4*dt*ux*w_ + 4*dt*v_*wy - 4*dt*vy*w_ + 2*pow(dt, 2)*u_*vx*wy
                                   - 2*pow(dt, 2)*u_*vy*wx - 2*pow(dt, 2)*ux*v_*wy + 2*pow(dt, 2)*ux*vy*w_
                                   + 2*pow(dt, 2)*uy*v_*wx - 2*pow(dt, 2)*uy*vx*w_)
                                / (4*dt*ux + 4*dt*vy + 4*dt*wz - 2*pow(dt, 2)*ux*vy + 2*pow(dt, 2)*uy*vx
                                   - 2*pow(dt, 2)*ux*wz + 2*pow(dt, 2)*uz*wx - 2*pow(dt, 2)*vy*wz + 2*pow(dt, 2)*vz*wy
                                   + pow(dt, 3)*ux*vy*wz - pow(dt, 3)*ux*vz*wy - pow(dt, 3)*uy*vx*wz
                                   + pow(dt, 3)*uy*vz*wx + pow(dt, 3)*uz*vx*wy - pow(dt, 3)*uz*vy*wx - 8);
            
    return displacement_meters;
}
