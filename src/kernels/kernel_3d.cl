#include "geography.cl"
#include "vector.cl"
#include "fields.cl"
#include "particle.cl"
#include "random.cl"
#include "advection_schemes.cl"
#include "eddy_diffusion.cl"
#include "windage.cl"

__kernel void advect(
    /* current vector field */
    __global const double *current_x,    // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int current_x_len,   // 1 <= current_x_len <= UINT_MAX + 1
    __global const double *current_y,    // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int current_y_len,   // 1 <= current_z_len <= UINT_MAX + 1
    __global const double *current_z,    // depth, meters, positive up
    const unsigned int current_z_len,   // 1 <= current_y_len <= UINT_MAX + 1
    __global const double *current_t,     // time, seconds since epoch, uniform spacing
    const unsigned int current_t_len,   // 1 <= current_t_len <= UINT_MAX + 1
    __global const float *current_U,    // m / s, 32 bit to save space
    __global const float *current_V,    // m / s
    __global const float *current_W,    // m / s
    /* wind vector field */
    __global const double *wind_x,    // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int wind_x_len,   // 1 <= wind_x_len <= UINT_MAX + 1
    __global const double *wind_y,    // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int wind_y_len,   // 1 <= wind_y_len <= UINT_MAX + 1
    __global const double *wind_t,     // time, seconds since epoch, uniform spacing
    const unsigned int wind_t_len,   // 1 <= wind_t_len <= UINT_MAX + 1
    __global const float *wind_U,    // m / s, 32 bit to save space
    __global const float *wind_V,    // m / s
    /* particle initialization information */
    __global const float *x0,         // lon, Deg E (-180 to 180)
    __global const float *y0,         // lat, Deg N (-90 to 90)
    __global const float *z0,         // depth, m, positive up, <= 0
    __global const double *release_date,         // unix timestamp
    /* advection time parameters */
    const double start_time,          // unix timestamp
    const double dt,             // seconds
    const unsigned int ntimesteps,
    const unsigned int save_every,
    /* output vectors */
    __global float *X_out,      // lon, Deg E (-180 to 180)
    __global float *Y_out,      // lat, Deg N (-90 to 90)
    __global float *Z_out,      // depth, m, positive up
    /* physics parameters */
    const unsigned int advection_scheme,
    const double eddy_diffusivity,
    const double windage_coeff)  // if nan, disables windage
{
    const unsigned int out_timesteps = ntimesteps / save_every;

    field3d current = {.x = current_x, .y = current_y, .z = current_z, .t = current_t,
                     .x_len = current_x_len, .y_len = current_y_len, .z_len = current_z_len, .t_len = current_t_len,
                     .x_spacing = (current_x[current_x_len-1]-current_x[0])/current_x_len,
                     .y_spacing = (current_y[current_y_len-1]-current_y[0])/current_y_len,
                     .t_spacing = (current_t[current_t_len-1]-current_t[0])/current_t_len,
                     .U = current_U, .V = current_V, .W = current_W};

    // turn 2d wind into 3d wind with singleton z
    double wind_z[1] = {0.0};
    field3d wind = {.x = wind_x, .y = wind_y, .z = (__global double *)wind_z, .t = wind_t,
                    .x_len = wind_x_len, .y_len = wind_y_len, .z_len = 1, .t_len = wind_t_len,
                    .x_spacing = (wind_x[wind_x_len-1]-wind_x[0])/wind_x_len,
                    .y_spacing = (wind_y[wind_y_len-1]-wind_y[0])/wind_y_len,
                    .t_spacing = (wind_t[wind_t_len-1]-wind_t[0])/wind_t_len,
                    .U = wind_U, .V = wind_V};

    // loop timesteps
    int global_id = get_global_id(0);
    particle p = {.id = global_id, .x = x0[global_id], .y = y0[global_id], .z = z0[global_id], .t = start_time};
    random_state rstate = {.a = ((unsigned int) p.id) + 1};  // for eddy diffusivity; must be unique across kernels, and nonzero.
    for (unsigned int timestep=0; timestep<ntimesteps; timestep++) {
        if (p.t < release_date[p.id]) {  // wait until the particle is released to start advecting and writing output
            p.t += dt;
            continue;
        }

        if (is_on_land(p, current)) {
            // do nothing; stuck forever
        } else {
            vector displacement_meters;
            if (advection_scheme == EULERIAN) {
                displacement_meters = eulerian_displacement(p, current, dt);
            } else if (advection_scheme == TAYLOR2) {
                displacement_meters = taylor2_displacement(p, current, dt);
            } else {
                return;  // can't throw errors but at least this way things will obviously fail
            }

            displacement_meters = add(displacement_meters, eddy_diffusion_meters(dt, &rstate, eddy_diffusivity));
            if (!isnan(windage_coeff)) {
                displacement_meters = add(displacement_meters, windage_meters(p, wind, dt, windage_coeff));
            }

            double dx_deg = meters_to_degrees_lon(displacement_meters.x, p.y);
            double dy_deg = meters_to_degrees_lat(displacement_meters.y, p.y);

            p = update_position_no_beaching(p, dx_deg, dy_deg, displacement_meters.z, current);
        }
        p.t += dt;
        // save if necessary
        if ((timestep+1) % save_every == 0) {
            unsigned int out_idx = (timestep+1)/save_every - 1;
            write_p(p, X_out, Y_out, Z_out, out_timesteps, out_idx);
        }
    }
}
