#include "wind_mixing_and_buoyancy.h"
#include "wind_driven_mixing.h"
#include "buoyancy.h"

vector wind_mixing_and_buoyancy_transport(
    particle p, field3d wind, field3d seawater_density,
    const double max_wave_height, const double wave_mixing_depth_factor,
    double dt, random_state *rstate, const bool wind_mixing_enabled) {
    /* p: the particle whose transport we're considering
     * wind: 10 meter wind vector field (m/s)
     * dt: the timestep (s)
     * rstate: random state for random turbulent mixing behavior
     * wind_mixing_enabled: flag to manually disable wind mixing, instead just return buoyancy transport.
     *
     * return: displacement (m) due to both wind mixing and particle buoyancy
     *  which form an equilibrium in the near-surface.  Outside of the near-surface, only buoyancy transport
     *  is considered.
     *  If density data cannot be found nearby the particle, flags failure by returning a vector with NAN components.
     */
    vector transport_meters = {.x = 0, .y = 0, .z = 0};

    double seawater_density_near_p = find_nearby_non_null_vector(p, seawater_density).x;
    if (isnan(seawater_density_near_p)) {
        vector failure = {.x = NAN, .y = NAN, .z = NAN};
        return failure;
    }
    double vertical_velocity = buoyancy_vertical_velocity(p.r, p.rho, p.CSF, seawater_density_near_p);

    vector nearest_wind = find_nearest_vector(p, wind, true);
    double wind_speed_10m = magnitude(nearest_wind);
    double MLD = mixed_layer_depth(wind_speed_10m, max_wave_height, wave_mixing_depth_factor);
    if (wind_mixing_enabled && (vertical_velocity >= 0) && (p.z > MLD)) {
        // under these conditions, in the near surface with floating particles,
        // a stochastic steady state is assumed, a balance between rising and turbulent mixing.
        // a sample can be pulled from this steady state; this is where the dynamics will take the particle.
        // From this, the displacement can be calculated.  A bit backwards, but still physically sound, assuming
        // that the model timestep is larger than the timescale of mixing (~hours).
        // Note the timestep doesn't even enter into this calculation, because of this assumption.
        double new_z = sample_concentration_profile(
            wind_speed_10m, vertical_velocity, max_wave_height, wave_mixing_depth_factor, rstate
        );

        transport_meters.z = new_z - p.z;
    } else {
        // outside of the very specific near-surface, rising/turbulence driven regime,
        // we just let the particle behave according to buoyancy.
        transport_meters.z = vertical_velocity * dt;
    }
    return transport_meters;
}
