#include "geography.cl"
#include "vector.cl"
#include "fields.cl"
#include "particle.cl"
#include "random.cl"
#include "advection_schemes.cl"
#include "diffusion.cl"
#include "windage.cl"
#include "gradients.cl"
#include "exit_codes.cl"
#include "vertical_profile.cl"

__kernel void advect(
    /* current vector field */
    __global const double *current_x,       // lon, Deg E (-180 to 180), uniform spacing, ascending,
    const unsigned int current_x_len,       // 1 <= current_x_len <= UINT_MAX + 1
    __global const double *current_y,       // lat, Deg N (-90 to 90), uniform spacing, ascending
    const unsigned int current_y_len,       // 1 <= current_y_len <= UINT_MAX + 1
    __global const double *current_t,       // time, seconds since epoch, uniform spacing, ascending
    const unsigned int current_t_len,       // 1 <= current_t_len <= UINT_MAX + 1
    __global const float *current_U,        // m / s, shape=(t, y, x) flattened, 32 bit to save space
    __global const float *current_V,        // m / s
    /* 10-meter wind vector field */
    __global const double *wind_x,          // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int wind_x_len,          // 1 <= wind_x_len <= UINT_MAX + 1
    __global const double *wind_y,          // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int wind_y_len,          // 1 <= wind_y_len <= UINT_MAX + 1
    __global const double *wind_t,          // time, seconds since epoch, uniform spacing
    const unsigned int wind_t_len,          // 1 <= wind_t_len <= UINT_MAX + 1
    __global const float *wind_U,           // m / s, shape=(t, y, x) flattened, 32 bit to save space
    __global const float *wind_V,           // m / s
    /* particle initialization */
    __global const float *x0,               // lon, Deg E (-180 to 180)
    __global const float *y0,               // lat, Deg N (-90 to 90)
    __global const double *release_date,    // unix timestamp
    /* physics */
    const unsigned int advection_scheme,
    const double windage_coefficient,  // if nan, disables windage
    const double eddy_diffusivity,    // m^2 s^-1
    /* advection time parameters */
    const double start_time,                // unix timestamp
    const double dt,                        // seconds
    const unsigned int ntimesteps,
    const unsigned int save_every,
    /* output vectors */
    __global float *X_out,                  // lon, Deg E (-180 to 180)
    __global float *Y_out,                  // lat, Deg N (-90 to 90)
    /* debugging */
    __global char *exit_code)
{
    const int global_id = get_global_id(0);
    if (exit_code[global_id] != SUCCESS) return;  // this indicates an error has already occured on this particle; quit

    const unsigned int out_timesteps = ntimesteps / save_every;

    /**** INITIALIZE STRUCTURES ****/
    field3d current = {
        .x = current_x, .y = current_y, .t = current_t,
        .x_len = current_x_len, .y_len = current_y_len, .t_len = current_t_len,
        .x_spacing = calculate_spacing(current_x, current_x_len),
        .y_spacing = calculate_spacing(current_y, current_y_len),
        .t_spacing = calculate_spacing(current_t, current_t_len),
        .U = current_U, .V = current_V,
    };
    current.x_is_circular = x_is_circular(current);

    field3d wind = {
        .x = wind_x, .y = wind_y, .t = wind_t,
        .x_len = wind_x_len, .y_len = wind_y_len, .t_len = wind_t_len,
        .x_spacing = calculate_spacing(wind_x, wind_x_len),
        .y_spacing = calculate_spacing(wind_y, wind_y_len),
        .t_spacing = calculate_spacing(wind_t, wind_t_len),
        .U = wind_U, .V = wind_V,
    };
    wind.x_is_circular = x_is_circular(wind);

    particle p = {
        .id = global_id,
        .x = x0[global_id],
        .y = y0[global_id],
        .t = start_time,
    };

    random_state rstate = {.a = ((unsigned int) p.id) + 1};  // for eddy diffusivity; must be unique across kernels, and nonzero.

    /*** LOOP OVER TIMESTEPS ***/
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

        if (!in_ocean_2d(p, current)) {
            // do nothing; stuck forever
        } else {
            vector displacement_meters;
            if (advection_scheme == EULERIAN) {
                displacement_meters = eulerian_displacement_2d(p, current, dt);
            } else if (advection_scheme == TAYLOR2) {
                displacement_meters = taylor2_displacement_2d(p, current, dt);
            } else {
                exit_code[global_id] = INVALID_ADVECTION_SCHEME;
                return;
            }

            displacement_meters = add(
                displacement_meters, eddy_diffusion_meters_2d(dt, &rstate, eddy_diffusivity)
            );
            if (!isnan(windage_coefficient)) {
                displacement_meters = add(
                    displacement_meters, explicit_windage(p, wind, dt, windage_coefficient)
                );
            }

            p = update_position_no_beaching_2d(p, displacement_meters, current);

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
            write_p_2d(p, X_out, Y_out, out_timesteps, out_idx);
        }
    }
    exit_code[global_id] = SUCCESS;
}
