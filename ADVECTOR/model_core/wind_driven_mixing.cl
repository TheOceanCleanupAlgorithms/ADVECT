#include "wind_driven_mixing.h"
#include "physical_constants.h"

double near_surface_diffusivity(double wind_speed_10m, const double max_wave_height);
double calculate_significant_wave_height(double wind_stress, const double max_wave_height);
double calculate_wind_stress(double wind_speed_10m);

double mixed_layer_depth(double wind_speed_10m, const double max_wave_height, const double wave_mixing_depth_factor) {
    double wind_stress = calculate_wind_stress(wind_speed_10m);
    double wave_height = calculate_significant_wave_height(wind_stress, max_wave_height);
    return -(wave_mixing_depth_factor * wave_height);
}

double sample_concentration_profile(
    double wind_speed_10m, double rise_velocity,
    const double max_wave_height, const double wave_mixing_depth_factor,
    random_state *rstate) {
    /* Generate a random depth, within mixed layer, PDF based on Kukulka 2012 eq. 4.*/
    // requirement: rise velocity > 0
    // This equation comes from normalizing eq. 4 from z=MLD to z=0 into a PDF, integrating it into a CDF,
    // then inverting this CDF so that it can be sampled using inverse transform sampling.
    // for 0 rise velocity, draw uniform depth in mixed layer (the inverted CDF asymptotes towards this behavior)
    double A_0 = near_surface_diffusivity(wind_speed_10m, max_wave_height);
    double MLD = mixed_layer_depth(wind_speed_10m, max_wave_height, wave_mixing_depth_factor);
    if (rise_velocity < 0) {
        return NAN;
    } else if (rise_velocity == 0) {
        return random_in_range(MLD, 0, rstate);
    } else {
        double w_b = rise_velocity;
        return A_0/w_b * log(
            exp(w_b/A_0 * MLD) +
            random(rstate) * (1 - exp(w_b/A_0 * MLD))
        );
    };
}

double near_surface_diffusivity(double wind_speed_10m, const double max_wave_height) {
    double wind_stress = calculate_wind_stress(wind_speed_10m);
    double significant_wave_height = calculate_significant_wave_height(wind_stress, max_wave_height);
    double frictional_water_velocity = sqrt(wind_stress/SURFACE_SEAWATER_DENSITY);  // Large and Pond (1981) eq. 2
    return frictional_water_velocity * VON_KARMAN_CONSTANT * 1.5 * significant_wave_height;
}

double calculate_significant_wave_height(double wind_stress, const double max_wave_height) {
    double frictional_air_velocity = sqrt(wind_stress/SURFACE_AIR_DENSITY);  // Large and Pond (1981) eq. 2
    return fmin(
        fabs(.96 / ACC_GRAVITY * pow(WAVE_AGE, 1.5) * pow(frictional_air_velocity, 2)),  // Kukulka 2012, just after eq. 3
        max_wave_height);  // the above equation generates unrealistically large waves for u10 > 20 m/s ish
}

double calculate_wind_stress(double wind_speed_10m) {
    // drag coefficient from Large and Pond 1981 eq. 19, extrapolated in both directions
    double C_D;
    if (wind_speed_10m <= 10) {  // m/s
        C_D = 1.14e-3;
    } else {
        C_D = .49e-3 + .065e-3 * wind_speed_10m;
    }
    return SURFACE_AIR_DENSITY * C_D * pow(wind_speed_10m, 2);  // SD Smith 1988 eq. 1
}
