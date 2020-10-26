#include "headers.cl"
#include "structs.cl"
#include "kernel_helpers.cl"
#include "advection_schemes.cl"
#include "eddy_diffusion.cl"

#define EULERIAN 0  // matches definitions in src/kernel_wrappers/Kernel2D.py
#define TAYLOR2 1

__kernel void advect(
    __global const double *field_x,    // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int x_len,   // <= UINT_MAX + 1
    __global const double *field_y,    // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int y_len,   // <= UINT_MAX + 1
    __global const double *field_t,     // time, seconds since epoch, uniform spacing
    const unsigned int t_len,   // <= UINT_MAX + 1
    __global const float *field_U,    // m / s, 32 bit to save space
    __global const float *field_V,    // m / s
    __global const float *x0,         // lon, Deg E (-180 to 180)
    __global const float *y0,         // lat, Deg N (-90 to 90)
    __global const double *release_date,         // unix timestamp
    const double start_time,          // unix timestamp
    const double dt,             // seconds
    const unsigned int ntimesteps,
    const unsigned int save_every,
    __global float *X_out,      // lon, Deg E (-180 to 180)
    __global float *Y_out,      // lat, Deg N (-90 to 90)
    const unsigned int advection_scheme,
    const double eddy_diffusivity)
{
    const unsigned int out_timesteps = ntimesteps / save_every;

    field2d field = {.x = field_x, .y = field_y, .t = field_t,
                     .x_len = x_len, .y_len = y_len, .t_len = t_len,
                     .x_spacing = field_x[1]-field_x[0],
                     .y_spacing = field_y[1]-field_y[0],
                     .t_spacing = field_t[1]-field_t[0],
                     .U = field_U, .V = field_V};

    // loop timesteps
    int global_id = get_global_id(0);
    particle p = {.id = global_id, .x = x0[global_id], .y = y0[global_id], .t = start_time};
    random_state rstate = {.a = ((unsigned int) p.id) + 1};  // for eddy diffusivity; must be unique across kernels, and nonzero.
    for (unsigned int timestep=0; timestep<ntimesteps; timestep++) {
        if (p.t < release_date[p.id]) {  // wait until the particle is released to start advecting and writing output
            p.t += dt;
            continue;
        }

        // find nearest neighbors in grid
        grid_point neighbor = find_nearest_neighbor(p, field);

        if (is_land(neighbor, field)) { // we're beached, me hearties!
            // do nothing; stays on land forever.
        } else {  // ah, the salt breeze, the open seas
            vector displacement_meters;
            if (advection_scheme == EULERIAN) {
                displacement_meters = eulerian_displacement(p, neighbor, field, dt);
            } else if (advection_scheme == TAYLOR2) {
                displacement_meters = taylor2_displacement(p, neighbor, field, dt);
            } else {
                return;  // can't throw errors but at least this way things will obviously fail
            }

            displacement_meters.x += eddy_diffusion_meters(dt, &rstate, eddy_diffusivity);
            displacement_meters.y += eddy_diffusion_meters(dt, &rstate, eddy_diffusivity);

            double dx_deg = meters_to_degrees_lon(displacement_meters.x, p.y);
            double dy_deg = meters_to_degrees_lat(displacement_meters.y, p.y);

            p = update_position(p, dx_deg, dy_deg);
        }
        p.t += dt;
        // save if necessary
        if ((timestep+1) % save_every == 0) {
            unsigned int out_idx = (timestep+1)/save_every - 1;
            write_p(p, X_out, Y_out, out_timesteps, out_idx);
        }
    }
}
