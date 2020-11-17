#ifndef GEOGRAPHY
#define GEOGRAPHY

#define INVALID_POSITION_LON 22.920583
#define INVALID_POSITION_LAT 11.831290

double degrees_lat_to_meters(double dy, double y);
double degrees_lon_to_meters(double dx, double y);
double meters_to_degrees_lon(double dx_meters, double y);
double meters_to_degrees_lat(double dy_meters, double y);

#endif // GEOGRAPHY
