#include "wind_mixing_and_buoyancy.h"
#include "wind_driven_mixing.h"
#include "buoyancy.h"

vector wind_mixing_and_buoyancy_transport(particle p, field3d wind, double dt, random_state *rstate, const bool wind_mixing_enabled) {
    vector transport_meters = {.x = 0, .y = 0, .z = 0};
    double vertical_velocity = buoyancy_vertical_velocity(p);
    if (isnan(vertical_velocity)) {
        transport_meters.z = NAN;  // pass on failure flag
        return transport_meters;
    }

    vector nearest_wind = index_vector_field(wind, find_nearest_neighbor(p, wind), true);
    double wind_speed_10m = magnitude(nearest_wind);

    if (wind_mixing_enabled && vertical_velocity > 0 && in_mixing_layer(p.z, wind_speed_10m)) {
        // under these conditions, in the near surface with floating particles,
        // a stochastic steady state is assumed, a balance between rising and turbulent mixing.
        // a sample can be pulled from this steady state; this is where the dynamics will take the particle.
        // From this, the displacement can be calculated.  A bit backwards, but still physically sound.
        // Note the timestep doesn't even enter into this calculation, because this balance is an equilibrium.
        double new_z = sample_concentration_profile(wind_speed_10m, vertical_velocity, rstate);
        transport_meters.z = new_z - p.z;
    } else {
        // outside of the very specific near-surface, rising/turbulence driven regime,
        // we just let the particle behave according to buoyancy.
        transport_meters.z = vertical_velocity * dt;
    }
    return transport_meters;
}
