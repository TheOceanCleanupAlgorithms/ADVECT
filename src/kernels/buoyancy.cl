#include "buoyancy.h"
#include "physical_constants.h"

double buoyancy_vertical_velocity(double radius, double density, double seawater_density) {
    /* calculate terminal velocity of a sphere with radius r and density rho
       due to density differential between particle and seawater,
       according to method in Dietrich 1982
       Density and kinematic viscosity of seawater are considered constant.
        
       radius: of sphere (m)
       density: of sphere (kg m^-3)
       seawater_density: of surrounding seawater (kg m^-3)
       
       return: terminal velocity of sphere due to buoyancy/drag force balance (positive up, m/s)
       If return is NAN, this indicates failure due to sphere radius being too large
        for this paramaterization.
    */
    double kinematic_viscosity_seawater = DYNAMIC_VISCOSITY_SEAWATER / seawater_density;
    double D_star = fabs((density - seawater_density) * ACC_GRAVITY * pow(2*radius, 3) /
                    (seawater_density * pow(kinematic_viscosity_seawater, 2)));

    double W_star;  // dimensionless settling velocity
    if (D_star < .05) {
        // Dietrich 1982, eq. 8
        W_star = pow(D_star, 2) / 5832;
    } else if (D_star <= 5e9) {
        // Dietrich 1982, eq. 9
        W_star = pow(10, -3.76715 +
                          1.92944 *     log10(D_star) -
                          0.09815 * pow(log10(D_star), 2) -
                          0.00575 * pow(log10(D_star), 3) +
                          0.00056 * pow(log10(D_star), 4));
    } else {
        return NAN;  // flag failure
    }

    // Dietrich 1982, eq. 5 rearranged
    double settling_velocity = cbrt((W_star * (seawater_density - density) * ACC_GRAVITY * kinematic_viscosity_seawater) / seawater_density);
    return -settling_velocity;
}
