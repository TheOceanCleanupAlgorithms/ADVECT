#include "geography.h"

// convert meters displacement to lat/lon and back(Reference: American Practical Navigator, Vol II, 1975 Edition, p 5)
double meters_to_degrees_lon(double dx_meters, double y) {
    double rlat = y * M_PI/180;
    if (cos(rlat) == 0)
	return 0;
    return dx_meters / (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

double meters_to_degrees_lat(double dy_meters, double y) {
    double rlat = y * M_PI/180;
    return dy_meters / (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}

double degrees_lon_to_meters(double dx, double y) {
    double rlat = y * M_PI/180;
    return dx * (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

double degrees_lat_to_meters(double dy, double y) {
    double rlat = y * M_PI/180;
    return dy * (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}
