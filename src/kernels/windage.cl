#include "headers.cl"
#include "structs.cl"
#include "kernel_helpers.cl"

vector windage_meters(particle p, field2d wind, double dt, double windage_coeff) {
    vector wind_displacement_meters = eulerian_displacement(p, find_nearest_neighbor(p, wind), wind, dt);
    wind_displacement_meters.x *= windage_coeff;
    wind_displacement_meters.y *= windage_coeff;
    return wind_displacement_meters;
}