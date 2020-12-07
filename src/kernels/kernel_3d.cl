#include "geography.cl"
#include "vector.cl"
#include "fields.cl"
#include "particle.cl"
#include "random.cl"
#include "advection_schemes.cl"
#include "eddy_diffusion.cl"
#include "windage.cl"
#include "buoyancy.cl"

enum ExitCode {SUCCESS = 0, NULL_LOCATION = 1, INVALID_LATITUDE = 2, PARTICLE_TOO_LARGE = 3,
               INVALID_ADVECTION_SCHEME = -1};
// positive codes are considered non-fatal, and are reported in outputfiles;
// negative codes are considered fatal, cause host-program termination, and are reserved for internal use.
// if you change these codes, update in src/kernel_wrappers/kernel_constants.py

__kernel void advect(
    /* current vector field */
    __global const double *current_x,       // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int current_x_len,       // 1 <= current_x_len <= UINT_MAX + 1
    __global const double *current_y,       // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int current_y_len,       // 1 <= current_y_len <= UINT_MAX + 1
    __global const double *current_z,       // depth, meters, positive up, sorted ascending
    const unsigned int current_z_len,       // 1 <= current_z_len <= UINT_MAX + 1
    __global const double *current_t,       // time, seconds since epoch, uniform spacing
    const unsigned int current_t_len,       // 1 <= current_t_len <= UINT_MAX + 1
    __global const float *current_U,        // m / s, shape=(t, z, y, x) flattened, 32 bit to save space
    __global const float *current_V,        // m / s
    __global const float *current_W,        // m / s
    /* wind vector field */
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
    __global const float *z0,               // depth, m, positive up, <= 0
    __global const double *release_date,    // unix timestamp
    __global const double *radius,          // particle radius, m
    __global const double *density,         // particle density, kg m^-3
    /* advection time parameters */
    const double start_time,                // unix timestamp
    const double dt,                        // seconds
    const unsigned int ntimesteps,
    const unsigned int save_every,
    /* output vectors */
    __global float *X_out,                  // lon, Deg E (-180 to 180)
    __global float *Y_out,                  // lat, Deg N (-90 to 90)
    __global float *Z_out,                  // depth, m, positive up
    /* physics */
    const unsigned int advection_scheme,
    const double eddy_diffusivity,
    const double windage_multiplier,  // if nan, disables windage
    /* debugging */
    __global char *exit_code)
{
    int global_id = get_global_id(0);
    if (exit_code[global_id] != SUCCESS) return;  // this indicates an error has already occured on this particle; quit

    const unsigned int out_timesteps = ntimesteps / save_every;

    field3d current = {.x = current_x, .y = current_y, .z = current_z, .t = current_t,
                     .x_len = current_x_len, .y_len = current_y_len, .z_len = current_z_len, .t_len = current_t_len,
                     .x_spacing = calculate_spacing(current_x, current_x_len),
                     .y_spacing = calculate_spacing(current_y, current_y_len),
                     .t_spacing = calculate_spacing(current_t, current_t_len),
                     .U = current_U, .V = current_V, .W = current_W,
                     .z_floor = calculate_coordinate_floor(current_z, current_z_len)};  // bottom edge of lowest layer

    // turn 2d wind into 3d wind with singleton z
    double wind_z[1] = {0.0};
    field3d wind = {.x = wind_x, .y = wind_y, .z = (__global double *)wind_z, .t = wind_t,
                    .x_len = wind_x_len, .y_len = wind_y_len, .z_len = 1, .t_len = wind_t_len,
                    .x_spacing = calculate_spacing(wind_x, wind_x_len),
                    .y_spacing = calculate_spacing(wind_y, wind_y_len),
                    .t_spacing = calculate_spacing(wind_t, wind_t_len),
                    .U = wind_U, .V = wind_V, .W = 0,
                    .z_floor = 0};

    // loop timesteps
    particle p = {.id = global_id, .r = radius[global_id], .rho = density[global_id],
                  .x = x0[global_id], .y = y0[global_id], .z = z0[global_id], .t = start_time};
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

        if (!in_ocean(p, current)) {
            // do nothing; stuck forever
        } else {
            vector displacement_meters;
            if (advection_scheme == EULERIAN) {
                displacement_meters = eulerian_displacement(p, current, dt);
            } else if (advection_scheme == TAYLOR2) {
                displacement_meters = taylor2_displacement(p, current, dt);
            } else {
                exit_code[global_id] = INVALID_ADVECTION_SCHEME;
                return;
            }

            vector buoyancy_transport_meters = buoyancy_transport(p, dt);
            if (isnan(buoyancy_transport_meters.z)) {
                exit_code[global_id] = PARTICLE_TOO_LARGE;
                return;
            }
            displacement_meters = add(displacement_meters, buoyancy_transport_meters);

            displacement_meters = add(displacement_meters, eddy_diffusion_meters(p.z, dt, &rstate, eddy_diffusivity));
            if (!isnan(windage_multiplier)) {
                displacement_meters = add(displacement_meters, windage_meters(p, wind, dt, windage_multiplier));
            }

            p = update_position_no_beaching(p, displacement_meters, current);

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
            write_p(p, X_out, Y_out, Z_out, out_timesteps, out_idx);
        }
    }
    exit_code[global_id] = SUCCESS;
}
