#include "wind_driven_mixing.h"
#include "physical_constants.h"
#include "diffusion.h"

double mixing_diffusivity(double z, double wind_speed_10m);

vector wind_mixing_meters(particle p, field3d wind_10m, const double dt, random_state *rstate) {
    vector nearest_wind_10m = index_vector_field(wind_10m, find_nearest_neighbor(p, wind_10m), true);
    double wind_speed_10m = sqrt(pow(nearest_wind_10m.x, 2) + pow(nearest_wind_10m.y, 2));
    double vertical_diffusivity = mixing_diffusivity(p.z, wind_speed_10m);
    double diffusion_amplitude = amplitude_of_diffusion(dt, 1, vertical_diffusivity);
    vector mixing_meters = {
        .x = 0,
        .y = 0,
        .z = random_within_magnitude(diffusion_amplitude, rstate)};
    return mixing_meters;
}


double mixing_diffusivity(double z, double wind_speed_10m) {
    double sigma = 35.0;  // Kukulka 2012 assumption: fully developed sea state
    // drag coefficient from Large and Pond 1981 eq. 19, extrapolated in both directions
    double C_D;
    if (wind_speed_10m <= 10) {  // m/s
        C_D = 1.14e-3;
    } else {
        C_D = .49e-3 + .065e-3 * wind_speed_10m;
    }
    double wind_stress = DENSITY_SURFACE_AIR * C_D * pow(wind_speed_10m, 2);  // Smith 1998 eq. 1
    double frictional_air_velocity = sqrt(wind_stress/DENSITY_SURFACE_AIR);
    double significant_wave_height = .96 / fabs(ACC_GRAVITY) * pow(sigma, 3/2) * pow(frictional_air_velocity, 2);
    if (z < -1.5*significant_wave_height) {  // deeper than turbulent layer
        return 0;
    } else {
        double frictional_water_velocity = sqrt(wind_stress/DENSITY_SEAWATER);
        return frictional_water_velocity * VON_KARMAN_CONSTANT * 1.5 * significant_wave_height;
    }
}
