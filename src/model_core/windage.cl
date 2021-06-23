#include "windage.h"

double calculate_windage_coeff(double r, double z);
double circular_segment_area(double R, double r);

vector explicit_windage(particle p, field3d wind_10m, double dt, double windage_coefficient) {
    /* Windage calculated based on a scaling coefficient that's explicitly provided */
    vector nearest_wind_10m = find_nearest_vector(p, wind_10m, true);
    vector nearest_wind_10cm = mul(nearest_wind_10m, wind_10cm_per_wind_10m);
    vector wind_displacement_meters = mul(nearest_wind_10cm, windage_coefficient * dt);
    return wind_displacement_meters;
}

vector windage_meters(particle p, field3d wind_10m, double dt, double windage_multiplier) {
    /* Physically-motivated windage, scaled by windage_multiplier.
     * Wind speed at 10cm is assumed to be representative of wind speed experienced by surfaced particles.
     */
    double windage_coeff = calculate_windage_coeff(p.r, p.z);
    return explicit_windage(p, wind_10m, dt, windage_coeff * windage_multiplier);
}

double calculate_windage_coeff(double r, double z) {
    /*  According to van der Mheen 2020, eq. 8 (originally Richardson 1997),
     *  Assumes z <= 0.
    */
    // first we have to calculate the ratio of emerged vs submerged area,
    // which is a lesson in the geometry of circles.
    double area_ratio;
    if (z == 0) {  // exactly half emerged; fast solution to likely the most common case
        area_ratio = 1;
    } else if (z <= -r) {  // fully submerged; fast solution to likely the second most common case
        area_ratio = 0;
    } else {                 // less than half emerged
        double submerged_segment = circular_segment_area(r, z);
        double emerged_segment = M_PI * pow(r, 2) - submerged_segment;
        area_ratio = emerged_segment / submerged_segment;
    }
    return sqrt(density_air_per_density_water * drag_ratio * area_ratio);
}

double circular_segment_area(double R, double r) {
    // R is the radius of the circle, r is the perpendicular distance between the
    // circle's center and the line which is intersecting the circle.  It follows that
    // this function is undefined for r > R.  Uses notation/equation from
    // Weisstein, Eric W. "Circular Segment." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CircularSegment.html
    // if r > 0, theta < 180 degrees, corresponding to a minor segment.
    // if r < 0, theta > 180 degrees, corresponding to a major segment.
    double theta = 2 * acos(r / R);  // this is the central angle of the segment
    return .5 * pow(R, 2) * (theta - sin(theta));
}
