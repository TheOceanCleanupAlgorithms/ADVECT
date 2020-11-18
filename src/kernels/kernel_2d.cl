#include "geography.cl"
#include "vector.cl"
#include "fields.cl"
#include "particle.cl"
#include "random.cl"
#include "advection_schemes.cl"
#include "eddy_diffusion.cl"
#include "windage.cl"

enum ExitCode {SUCCESS = 0, FAILURE = 1, INVALID_ADVECTION_SCHEME = 2, NULL_LOCATION = 3, INVALID_LATITUDE = 4};
// if you change these codes, update in src/kernel_wrappers/ExitCodes.py

__kernel void advect(
    /* current vector field */
    __global const double *current_x,    // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int current_x_len,   // 1 <= current_x_len <= UINT_MAX + 1
    __global const double *current_y,    // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int current_y_len,   // 1 <= current_y_len <= UINT_MAX + 1
    __global const double *current_t,     // time, seconds since epoch, uniform spacing
    const unsigned int current_t_len,   // 1 <= current_t_len <= UINT_MAX + 1
    __global const float *current_U,    // m / s, 32 bit to save space
    __global const float *current_V,    // m / s
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
    __global const double *release_date,         // unix timestamp
    /* advection time parameters */
    const double start_time,          // unix timestamp
    const double dt,             // seconds
    const unsigned int ntimesteps,
    const unsigned int save_every,
    /* output vectors */
    __global float *X_out,      // lon, Deg E (-180 to 180)
    __global float *Y_out,      // lat, Deg N (-90 to 90)
    /* physics parameters */
    const unsigned int advection_scheme,
    const double eddy_diffusivity,
    const double windage_coeff,  // if nan, disables windage
    /* debugging utility */
    __global unsigned char *exit_code)
{
    int global_id = get_global_id(0);
    if (exit_code[global_id] != SUCCESS) return;  // this indicates an error has already occured on this particle; quit

    const unsigned int out_timesteps = ntimesteps / save_every;

    field2d current = {.x = current_x, .y = current_y, .t = current_t,
                     .x_len = current_x_len, .y_len = current_y_len, .t_len = current_t_len,
                     .x_spacing = (current_x[current_x_len-1]-current_x[0])/current_x_len,
                     .y_spacing = (current_y[current_y_len-1]-current_y[0])/current_y_len,
                     .t_spacing = (current_t[current_t_len-1]-current_t[0])/current_t_len,
                     .U = current_U, .V = current_V};

    field2d wind = {.x = wind_x, .y = wind_y, .t = wind_t,
                    .x_len = wind_x_len, .y_len = wind_y_len, .t_len = wind_t_len,
                    .x_spacing = (wind_x[wind_x_len-1]-wind_x[0])/wind_x_len,
                    .y_spacing = (wind_y[wind_y_len-1]-wind_y[0])/wind_y_len,
                    .t_spacing = (wind_t[wind_t_len-1]-wind_t[0])/wind_t_len,
                    .U = wind_U, .V = wind_V};

    // loop timesteps
    particle p = {.id = global_id, .x = x0[global_id], .y = y0[global_id], .t = start_time};
    random_state rstate = {.a = ((unsigned int) p.id) + 1};  // for eddy diffusivity; must be unique across kernels, and nonzero.
    for (unsigned int timestep=0; timestep<ntimesteps; timestep++) {
        if (p.t < release_date[p.id]) {  // wait until the particle is released to start advecting and writing output
            p.t += dt;
            continue;
        }

        // quit if particle has null location, this is a disallowed state
        if (isnan(p.x) || isnan(p.y)) {
            exit_code[global_id] = NULL_LOCATION;
            return;
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
                exit_code[global_id] = INVALID_ADVECTION_SCHEME;
                return;  // complete execution
            }

            displacement_meters = add(displacement_meters, eddy_diffusion_meters(dt, &rstate, eddy_diffusivity));
            if (!isnan(windage_coeff)) {
                displacement_meters = add(displacement_meters, windage_meters(p, wind, dt, windage_coeff));
            }

            double dx_deg = meters_to_degrees_lon(displacement_meters.x, p.y);
            double dy_deg = meters_to_degrees_lat(displacement_meters.y, p.y);

            p = update_position_no_beaching(p, dx_deg, dy_deg, current);

            // If, for some reason, the particle latitude goes completely out of [-90, 90], note the error and exit.
            if (fabs(p.y) > 90) {
                exit_code[global_id] = INVALID_LATITUDE;
                return;
            }
        }
        p.t += dt;
        // save if necessary
        if ((timestep+1) % save_every == 0) {
            unsigned int out_idx = (timestep+1)/save_every - 1;
            write_p(p, X_out, Y_out, out_timesteps, out_idx);
        }
    }
    exit_code[global_id] = SUCCESS;
}
