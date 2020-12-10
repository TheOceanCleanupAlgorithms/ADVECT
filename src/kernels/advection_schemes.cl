#include "advection_schemes.h"
#include "gradients.h"

vector taylor2_formula(vector V, vector V_x, vector V_y, vector V_z, vector V_t, double dt);

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

    // calculate gradients
    vector V_x = x_partial(p, field);
    vector V_y = y_partial(p, field);
    vector V_z = z_partial(p, field);
    vector V_t = t_partial(p, field);
    
    // nan means undefined gradient, due to outside domain; use 0
    V_x.x = isnan(V_x.x) ? 0 : V_x.x;
    V_x.y = isnan(V_x.y) ? 0 : V_x.y;
    V_x.z = isnan(V_x.z) ? 0 : V_x.z;
    V_y.x = isnan(V_y.x) ? 0 : V_y.x;
    V_y.y = isnan(V_y.y) ? 0 : V_y.y;
    V_y.z = isnan(V_y.z) ? 0 : V_y.z;
    V_z.x = isnan(V_z.x) ? 0 : V_z.x;
    V_z.y = isnan(V_z.y) ? 0 : V_z.y;
    V_z.z = isnan(V_z.z) ? 0 : V_z.z;
    V_t.x = isnan(V_t.x) ? 0 : V_t.x;
    V_t.y = isnan(V_t.y) ? 0 : V_t.y;
    V_t.z = isnan(V_t.z) ? 0 : V_t.z;

    // calculate formula
    return taylor2_formula(V, V_x, V_y, V_z, V_t, dt);
}

vector taylor2_formula(vector V, vector V_x, vector V_y, vector V_z, vector V_t, double dt) {
    /* advect particle using second-order taylor approx advection scheme
     * simplified form of scheme in Appendix A4 of Tim Jensen Master's Thesis; copied from his code
     * at https://github.com/TimJansen94/3D-dispersal-model-Tim/blob/master/run_model/funcs_advect.m
     * V: velocity at particle's location (m / s)
     * V_x: partial derivative w.r.t. x at particle's location (1 / s)
     * V_y: partial derivative w.r.t. y at particle's location (1 / s)
     * V_z: partial derivative w.r.t. z at particle's location (1 / s)
     * V_t: partial derivative w.r.t. t at particle's location (m / s^2)
     * dt: timestep (s)
     */

    // simplifying terms
    double u_ = V.x + (dt*V_t.x)/2;
    double v_ = V.y + (dt*V_t.y)/2;
    double w_ = V.z + (dt*V_t.z)/2;
    
    vector displacement_meters;
    displacement_meters.x = dt * -(8*u_ - 4*dt*u_*V_y.y + 4*dt*V_y.x*v_ - 4*dt*u_*V_z.z + 4*dt*V_z.x*w_ + 2*pow(dt, 2)*u_*V_y.y*V_z.z
                                   - 2*pow(dt, 2)*u_*V_z.y*V_y.z - 2*pow(dt, 2)*V_y.x*v_*V_z.z + 2*pow(dt, 2)*V_y.x*V_z.y*w_
                                   + 2*pow(dt, 2)*V_z.x*v_*V_y.z - 2*pow(dt, 2)*V_z.x*V_y.y*w_)
                                / (4*dt*V_x.x + 4*dt*V_y.y + 4*dt*V_z.z - 2*pow(dt, 2)*V_x.x*V_y.y + 2*pow(dt, 2)*V_y.x*V_x.y
                                   - 2*pow(dt, 2)*V_x.x*V_z.z + 2*pow(dt, 2)*V_z.x*V_x.z - 2*pow(dt, 2)*V_y.y*V_z.z + 2*pow(dt, 2)*V_z.y*V_y.z
                                   + pow(dt, 3)*V_x.x*V_y.y*V_z.z - pow(dt, 3)*V_x.x*V_z.y*V_y.z - pow(dt, 3)*V_y.x*V_x.y*V_z.z
                                   + pow(dt, 3)*V_y.x*V_z.y*V_x.z + pow(dt, 3)*V_z.x*V_x.y*V_y.z - pow(dt, 3)*V_z.x*V_y.y*V_x.z - 8);
    displacement_meters.y = dt * -(8*v_ + 4*dt*u_*V_x.y - 4*dt*V_x.x*v_ - 4*dt*v_*V_z.z + 4*dt*V_z.y*w_ - 2*pow(dt, 2)*u_*V_x.y*V_z.z
                                   + 2*pow(dt, 2)*u_*V_z.y*V_x.z + 2*pow(dt, 2)*V_x.x*v_*V_z.z - 2*pow(dt, 2)*V_x.x*V_z.y*w_
                                   - 2*pow(dt, 2)*V_z.x*v_*V_x.z + 2*pow(dt, 2)*V_z.x*V_x.y*w_)
                                / (4*dt*V_x.x + 4*dt*V_y.y + 4*dt*V_z.z - 2*pow(dt, 2)*V_x.x*V_y.y + 2*pow(dt, 2)*V_y.x*V_x.y
                                   - 2*pow(dt, 2)*V_x.x*V_z.z + 2*pow(dt, 2)*V_z.x*V_x.z - 2*pow(dt, 2)*V_y.y*V_z.z + 2*pow(dt, 2)*V_z.y*V_y.z
                                   + pow(dt, 3)*V_x.x*V_y.y*V_z.z - pow(dt, 3)*V_x.x*V_z.y*V_y.z - pow(dt, 3)*V_y.x*V_x.y*V_z.z
                                   + pow(dt, 3)*V_y.x*V_z.y*V_x.z + pow(dt, 3)*V_z.x*V_x.y*V_y.z - pow(dt, 3)*V_z.x*V_y.y*V_x.z - 8);
    displacement_meters.z = dt * -(8*w_ + 4*dt*u_*V_x.z - 4*dt*V_x.x*w_ + 4*dt*v_*V_y.z - 4*dt*V_y.y*w_ + 2*pow(dt, 2)*u_*V_x.y*V_y.z
                                   - 2*pow(dt, 2)*u_*V_y.y*V_x.z - 2*pow(dt, 2)*V_x.x*v_*V_y.z + 2*pow(dt, 2)*V_x.x*V_y.y*w_
                                   + 2*pow(dt, 2)*V_y.x*v_*V_x.z - 2*pow(dt, 2)*V_y.x*V_x.y*w_)
                                / (4*dt*V_x.x + 4*dt*V_y.y + 4*dt*V_z.z - 2*pow(dt, 2)*V_x.x*V_y.y + 2*pow(dt, 2)*V_y.x*V_x.y
                                   - 2*pow(dt, 2)*V_x.x*V_z.z + 2*pow(dt, 2)*V_z.x*V_x.z - 2*pow(dt, 2)*V_y.y*V_z.z + 2*pow(dt, 2)*V_z.y*V_y.z
                                   + pow(dt, 3)*V_x.x*V_y.y*V_z.z - pow(dt, 3)*V_x.x*V_z.y*V_y.z - pow(dt, 3)*V_y.x*V_x.y*V_z.z
                                   + pow(dt, 3)*V_y.x*V_z.y*V_x.z + pow(dt, 3)*V_z.x*V_x.y*V_y.z - pow(dt, 3)*V_z.x*V_y.y*V_x.z - 8);
    return displacement_meters;
}
