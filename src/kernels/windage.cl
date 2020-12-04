#include "windage.h"
#include "advection_schemes.h"

#define density_ratio 1.17e-3  // density of air / density of water (van der Mheen 2020 near eq. 8)
#define drag_ratio 1           // drag coefficient in air / drag coefficient in water

double calculate_windage_coeff(double r, double z);
double circular_segment_area(double R, double r);

vector windage_meters(particle p, field3d wind, double dt, double windage_multiplier) {
    /* Physically-motivated windage, scaled by windage_multiplier */
    double windage_coeff = calculate_windage_coeff(p.r, p.z);
    vector wind_displacement_meters = eulerian_displacement(p, wind, dt);
    wind_displacement_meters.x *= windage_coeff * windage_multiplier;
    wind_displacement_meters.y *= windage_coeff * windage_multiplier;
    wind_displacement_meters.z = 0;
    return wind_displacement_meters;
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
    } else if (-z >= r) {  // fully submerged; fast solution to likely the second most common case
        area_ratio = 0;
    } else {                 // less than half emerged
        double submerged_segment = circular_segment_area(r, z);
        double emerged_segment = M_PI * pow(r, 2) - submerged_segment;
        area_ratio = emerged_segment / submerged_segment;
    }
    return sqrt(density_ratio * drag_ratio * area_ratio);
}

double circular_segment_area(double R, double r) {
    // R is the radius of the circle, r is the perpendicular distance between the
    // circle's center and the line which is intersecting the circle.  It follows that
    // this function is undefined for r > R.  Uses notation/equation from
    // Weisstein, Eric W. "Circular Segment." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CircularSegment.html
    // r > 0 corresponds to a minor segment, r < 0 corresponds to a major segment
    double theta = 2 * acos(r / R);  // this is the central angle of the segment
    return .5 * pow(R, 2) * (theta - sin(theta));
}
