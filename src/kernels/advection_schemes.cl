#include "advection_schemes.h"
#include "geography.h"
#include "gradients.h"

vector eulerian_displacement(particle p, field2d field, double dt) {
    // find nearest neighbors in grid
    grid_point neighbor = find_nearest_neighbor(p, field);
    // find U and V nearest to particle position
    vector V = index_vector_field(field, neighbor, true);

    //////////// advect particle using euler forward advection scheme
    // meters displacement
    vector displacement_meters = {.x = V.x * dt,
                                  .y = V.y * dt};
    return displacement_meters;
}

vector taylor2_displacement(particle p, field2d field, double dt) {
    grid_point gp = find_nearest_neighbor(p, field);
    vector V = index_vector_field(field, gp, true);

    // Calculate gradients
    // nan result means p outside domain
    vector V_x = x_partial(p, field);
    vector V_y = y_partial(p, field);
    vector V_t = t_partial(p, field);
    V_x.x = isnan(V_x.x) ? 0 : V_x.x;
    V_x.y = isnan(V_x.y) ? 0 : V_x.y;
    V_y.x = isnan(V_y.x) ? 0 : V_y.x;
    V_y.y = isnan(V_y.y) ? 0 : V_y.y;
    V_t.x = isnan(V_t.x) ? 0 : V_t.x;
    V_t.y = isnan(V_t.y) ? 0 : V_t.y;

    // simplifying term
    double u_ = V.x + (dt*V_t.x)/2;
    double v_ = V.y + (dt*V_t.y)/2;

    //////////// advect particle using second-order taylor approx advection scheme (Black and Gay, 1990, eq. 12/13)
    vector displacement_meters;
    displacement_meters.x = (u_ + (V_y.x*v_ - V_y.y*u_) * dt/2) * dt / ((1 - V_x.x*dt/2) * (1 - V_y.y*dt/2) - (V_y.x*V_x.y * pow(dt, 2)) / 4);
    displacement_meters.y = (v_ + (V_x.y*u_ - V_x.x*v_) * dt/2) * dt / ((1 - V_x.x*dt/2) * (1 - V_y.y*dt/2) - (V_y.x*V_x.y * pow(dt, 2)) / 4);

    return displacement_meters;
}
