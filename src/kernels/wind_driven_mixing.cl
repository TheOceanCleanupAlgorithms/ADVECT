#include "wind_driven_mixing.h"
#include "physical_constants.h"
#include "random.h"

double near_surface_diffusivity(double wind_speed_10m);
double calculate_significant_wave_height(double wind_stress);
double calculate_wind_stress(double wind_speed_10m);

bool in_mixing_layer(double z, double wind_speed_10m) {
    double wind_stress = calculate_wind_stress(wind_speed_10m);
    double wave_height = calculate_significant_wave_height(wind_stress);
    return z > -1.5 * wave_height;  // definition from kukulka 2012
}


double sample_concentration_profile(double wind_speed_10m, double rise_velocity, random_state *rstate) {
    /* Generate a random depth, weighted by Kukulka 2012 eq. 4.*/
    // This equation comes from normalizing eq. 4 into a PDF, integrating it into a CDF,
    // then inverting this CDF so that it can be sampled using inverse transform sampling.
    double A_0 = near_surface_diffusivity(wind_speed_10m);
    random(rstate);
    return (A_0 / rise_velocity * log(random(rstate)));
}


double near_surface_diffusivity(double wind_speed_10m) {
    double wind_stress = calculate_wind_stress(wind_speed_10m);
    double significant_wave_height = calculate_significant_wave_height(wind_stress);
    double frictional_water_velocity = sqrt(wind_stress/DENSITY_SEAWATER);  // Large and Pond (1981) eq. 2
    return frictional_water_velocity * VON_KARMAN_CONSTANT * 1.5 * significant_wave_height;
}


double calculate_significant_wave_height(double wind_stress) {
    const double wave_age = 35.0;  // Kukulka 2012 assumption: fully developed sea state
    double frictional_air_velocity = sqrt(wind_stress/DENSITY_SURFACE_AIR);  // Large and Pond (1981) eq. 2
    return fabs(.96 / ACC_GRAVITY * pow(wave_age, 1.5) * pow(frictional_air_velocity, 2));  // Kukulka 2012, just after eq. 3
}


double calculate_wind_stress(double wind_speed_10m) {
    // drag coefficient from Large and Pond 1981 eq. 19, extrapolated in both directions
    double C_D;
    if (wind_speed_10m <= 10) {  // m/s
        C_D = 1.14e-3;
    } else {
        C_D = .49e-3 + .065e-3 * wind_speed_10m;
    }
    return DENSITY_SURFACE_AIR * C_D * pow(wind_speed_10m, 2);  // Smith 1998 eq. 1
}
