#include "geography.cl"
#include "vector.cl"
#include "fields.cl"
#include "particle.cl"
#include "random.cl"
#include "advection_schemes.cl"
#include "diffusion.cl"
#include "windage.cl"
#include "buoyancy.cl"
#include "gradients.cl"
#include "vertical_profile.cl"
#include "wind_driven_mixing.cl"
#include "wind_mixing_and_buoyancy.cl"
#include "exit_codes.cl"

__kernel void advect(
    /* current vector field */
    __global const double *current_x,       // lon, Deg E (-180 to 180), uniform spacing, ascending,
    const unsigned int current_x_len,       // 1 <= current_x_len <= UINT_MAX + 1
    __global const double *current_y,       // lat, Deg N (-90 to 90), uniform spacing, ascending
    const unsigned int current_y_len,       // 1 <= current_y_len <= UINT_MAX + 1
    __global const double *current_z,       // depth, meters, positive up, sorted ascending
    const unsigned int current_z_len,       // 1 <= current_z_len <= UINT_MAX + 1
    __global const double *current_t,       // time, seconds since epoch, uniform spacing, ascending
    const unsigned int current_t_len,       // 1 <= current_t_len <= UINT_MAX + 1
    __global const float *current_U,        // m / s, shape=(t, z, y, x) flattened, 32 bit to save space
    __global const float *current_V,        // m / s
    __global const float *current_W,        // m / s
    __global const float *current_bathy,    // m, positive up, shape=(y, x) flattened
    /* 10-meter wind vector field */
    __global const double *wind_x,          // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int wind_x_len,          // 1 <= wind_x_len <= UINT_MAX + 1
    __global const double *wind_y,          // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int wind_y_len,          // 1 <= wind_y_len <= UINT_MAX + 1
    __global const double *wind_t,          // time, seconds since epoch, uniform spacing
    const unsigned int wind_t_len,          // 1 <= wind_t_len <= UINT_MAX + 1
    __global const float *wind_U,           // m / s, shape=(t, y, x) flattened, 32 bit to save space
    __global const float *wind_V,           // m / s
    /* seawater density field */
    __global const double *seawater_density_x,       // lon, Deg E (-180 to 180), uniform spacing, ascending,
    const unsigned int seawater_density_x_len,       // 1 <= seawater_density_x_len <= UINT_MAX + 1
    __global const double *seawater_density_y,       // lat, Deg N (-90 to 90), uniform spacing, ascending
    const unsigned int seawater_density_y_len,       // 1 <= seawater_density_y_len <= UINT_MAX + 1
    __global const double *seawater_density_z,       // depth, meters, positive up, sorted ascending
    const unsigned int seawater_density_z_len,       // 1 <= seawater_density_z_len <= UINT_MAX + 1
    __global const double *seawater_density_t,       // time, seconds since epoch, sorted ascending
    const unsigned int seawater_density_t_len,       // 1 <= seawater_density_t_len <= UINT_MAX + 1
    __global const float *seawater_density_values,        // kg m^-3, shape=(t, z, y, x) flattened, 32 bit to save space
    /* particle initialization */
    __global const float *x0,               // lon, Deg E (-180 to 180)
    __global const float *y0,               // lat, Deg N (-90 to 90)
    __global const float *z0,               // depth, m, positive up, <= 0
    __global const double *release_date,    // unix timestamp
    __global const double *radius,          // particle radius, m
    __global const double *density,         // particle density, kg m^-3
    __global const double *corey_shape_factor,  // particle shape factor, unitless, must be in (.15, 1]
    /* physics */
    const unsigned int advection_scheme,
    const double windage_multiplier,  // if nan, disables windage
    const unsigned int wind_mixing_enabled,   // 0: false, 1:true; toggle near-surface wind mixing
    const double max_wave_height,     // caps parameterization of significant wave height based on wind speed
    const double wave_mixing_depth_factor,  // max mixing depth = wave_mixing_depth_factor * significant wave height
    /* eddy diffusivity */
    __global const double *horizontal_eddy_diffusivity_z,  // depth coordinates, m, positive up, sorted ascending
    __global const double *horizontal_eddy_diffusivity_values,    // m^2 s^-1
    const unsigned int horizontal_eddy_diffusivity_len,
    __global const double *vertical_eddy_diffusivity_z,  // depth coordinates, m, positive up, sorted ascending
    __global const double *vertical_eddy_diffusivity_values,    // m^2 s^-1
    const unsigned int vertical_eddy_diffusivity_len,
    /* advection time parameters */
    const double start_time,                // unix timestamp
    const double dt,                        // seconds
    const unsigned int ntimesteps,
    const unsigned int save_every,
    /* output vectors */
    __global float *X_out,                  // lon, Deg E (-180 to 180)
    __global float *Y_out,                  // lat, Deg N (-90 to 90)
    __global float *Z_out,                  // depth, m, positive up
    /* debugging */
    __global char *exit_code)
{
    const int global_id = get_global_id(0);
    if (exit_code[global_id] != SUCCESS) return;  // this indicates an error has already occured on this particle; quit

    const unsigned int out_timesteps = ntimesteps / save_every;

    /**** INITIALIZE STRUCTURES ****/
    field3d current = {.x = current_x, .y = current_y, .z = current_z, .t = current_t,
                     .x_len = current_x_len, .y_len = current_y_len, .z_len = current_z_len, .t_len = current_t_len,
                     .x_spacing = calculate_spacing(current_x, current_x_len),
                     .y_spacing = calculate_spacing(current_y, current_y_len),
                     .t_spacing = calculate_spacing(current_t, current_t_len),
                     .U = current_U, .V = current_V, .W = current_W,
                     .bathy = current_bathy};
    current.x_is_circular = x_is_circular(current);

    field3d wind = {.x = wind_x, .y = wind_y, .t = wind_t,
                    .x_len = wind_x_len, .y_len = wind_y_len, .t_len = wind_t_len,
                    .x_spacing = calculate_spacing(wind_x, wind_x_len),
                    .y_spacing = calculate_spacing(wind_y, wind_y_len),
                    .t_spacing = calculate_spacing(wind_t, wind_t_len),
                    .U = wind_U, .V = wind_V};
    wind.x_is_circular = x_is_circular(wind);

    field3d seawater_density = {
        .x = seawater_density_x, .y = seawater_density_y, .z = seawater_density_z, .t = seawater_density_t,
        .x_len = seawater_density_x_len, .y_len = seawater_density_y_len, .z_len = seawater_density_z_len, .t_len = seawater_density_t_len,
        .x_spacing = calculate_spacing(seawater_density_x, seawater_density_x_len),
        .y_spacing = calculate_spacing(seawater_density_y, seawater_density_y_len),
        .t_spacing = NAN,
        .U = seawater_density_values,
    };
    seawater_density.x_is_circular = x_is_circular(seawater_density);

    vertical_profile horizontal_eddy_diffusivity_profile = {
        .values = horizontal_eddy_diffusivity_values,
        .z = horizontal_eddy_diffusivity_z,
        .len = horizontal_eddy_diffusivity_len};

    vertical_profile vertical_eddy_diffusivity_profile = {
        .values = vertical_eddy_diffusivity_values,
        .z = vertical_eddy_diffusivity_z,
        .len = vertical_eddy_diffusivity_len};

    particle p = {
        .id = global_id, .r = radius[global_id], .rho = density[global_id], .CSF = corey_shape_factor[global_id],
        .x = x0[global_id], .y = y0[global_id], .z = z0[global_id], .t = start_time};

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

            displacement_meters = add(displacement_meters, eddy_diffusion_meters(p.z, dt, &rstate,
                                                                                 horizontal_eddy_diffusivity_profile,
                                                                                 vertical_eddy_diffusivity_profile));
            if (!isnan(windage_multiplier)) {
                displacement_meters = add(displacement_meters, windage_meters(p, wind, dt, windage_multiplier));
            }

            vector wind_mixing_and_buoyancy = wind_mixing_and_buoyancy_transport(
                p, wind, seawater_density, max_wave_height, wave_mixing_depth_factor, dt, &rstate, wind_mixing_enabled
            );
            if (isnan(wind_mixing_and_buoyancy.x) && isnan(wind_mixing_and_buoyancy.y) && isnan(wind_mixing_and_buoyancy.z)) {
                exit_code[global_id] = SEAWATER_DENSITY_LOOKUP_FAILURE;
                return;
            }
            displacement_meters = add(displacement_meters, wind_mixing_and_buoyancy);

            p = update_position_no_beaching_3d(p, displacement_meters, current);

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
