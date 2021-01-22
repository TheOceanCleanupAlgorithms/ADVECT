#include "wind_mixing_and_buoyancy.h"
#include "wind_driven_mixing.h"
#include "buoyancy.h"

vector wind_mixing_and_buoyancy_transport(particle p, field3d wind, vertical_profile density_profile, double dt, random_state *rstate, const bool wind_mixing_enabled) {
    /* p: the particle whose transport we're considering
     * wind: 10 meter wind vector field (m/s)
     * dt: the timestep (s)
     * rstate: random state for random turbulent mixing behavior
     * wind_mixing_enabled: flag to manually disable wind mixing, instead just return buoyancy transport.
     *
     * return: displacement (m) due to both wind mixing and particle buoyancy
     *  which form an equilibrium in the near-surface.  Outside of the near-surface, only buoyancy transport
     *  is considered.
     *  If return is NAN, this means the particle radius is outside valid domain for the buoyancy transport paramaterization.
     */
    vector transport_meters = {.x = 0, .y = 0, .z = 0};

    double seawater_density = sample_profile(density_profile, p.z);
    double vertical_velocity = buoyancy_vertical_velocity(p.r, p.rho, p.CSF, seawater_density);
    if (isnan(vertical_velocity)) {
        transport_meters.z = NAN;  // pass on failure flag
        return transport_meters;
    }

    vector nearest_wind = find_nearest_vector(p, wind, true);
    double wind_speed_10m = magnitude(nearest_wind);
    if (wind_mixing_enabled && (vertical_velocity >= 0) && (p.z > mixed_layer_depth(wind_speed_10m))) {
        // under these conditions, in the near surface with floating particles,
        // a stochastic steady state is assumed, a balance between rising and turbulent mixing.
        // a sample can be pulled from this steady state; this is where the dynamics will take the particle.
        // From this, the displacement can be calculated.  A bit backwards, but still physically sound, assuming
        // that the model timestep is larger than the timescale of mixing (~hours).
        // Note the timestep doesn't even enter into this calculation, because of this assumption.
        double new_z = sample_concentration_profile(wind_speed_10m, vertical_velocity, rstate);

        transport_meters.z = new_z - p.z;
    } else {
        // outside of the very specific near-surface, rising/turbulence driven regime,
        // we just let the particle behave according to buoyancy.
        transport_meters.z = vertical_velocity * dt;
    }
    return transport_meters;
}
