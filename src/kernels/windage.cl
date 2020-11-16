#include "windage.h"
#include "advection_schemes.h"

vector windage_meters(particle p, field3d wind, double dt, double windage_coeff) {
    if (p.z < MINIMUM_WINDAGE_DEPTH) {
        vector no_displacement = {.x = 0, .y = 0};
        return no_displacement;
    }
    vector wind_displacement_meters = eulerian_displacement(p, wind, dt);
    wind_displacement_meters.x *= windage_coeff;
    wind_displacement_meters.y *= windage_coeff;
    return wind_displacement_meters;
}
