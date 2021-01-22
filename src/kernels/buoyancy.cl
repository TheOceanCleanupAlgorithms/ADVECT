#include "buoyancy.h"
#include "physical_constants.h"

double dimensionless_particle_diameter(double radius, double density, double seawater_density, double kinematic_viscosity_seawater);
double dimensionless_settling_velocity(double D_star, double CSF);

double buoyancy_vertical_velocity(double radius, double density, double corey_shape_factor, double seawater_density) {
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
    double D_star = dimensionless_particle_diameter(radius, density, seawater_density, kinematic_viscosity_seawater);

    double W_star = dimensionless_settling_velocity(D_star, corey_shape_factor);

    // Dietrich 1982, eq. 5 rearranged
    double settling_velocity = cbrt((W_star * (seawater_density - density) * ACC_GRAVITY * kinematic_viscosity_seawater) / seawater_density);
    return -settling_velocity;
}

double dimensionless_particle_diameter(double radius, double density, double seawater_density, double kinematic_viscosity_seawater) {
    /* radius: particle radius (m)
     * density: particle density (kg m^-3)
     * seawater_density: (kg m^-3)
     * kinematic_viscosity_seawater: (m^2 s^-1)
     * returns: dimensionless particle diameter, according to Dietrich eq. 6
     */
    return fabs((density - seawater_density) * ACC_GRAVITY * pow(2*radius, 3) /
                    (seawater_density * pow(kinematic_viscosity_seawater, 2)));
}

double dimensionless_settling_velocity(double D_star, double CSF) {
    /* According to Dietrich 1982 eq. 8/9
     * D_star: dimensionless particle diameter, Dietrich 1982
     * CSF: Corey Shape Factor, domain [.15, 1], per Dietrich 1982.
     * returns: dimensionless settling velocity.  If NAN, D_star was outside domain (aka particle too big)
     */
    double R_1;  // this coefficient predicts settling velocity for a perfect sphere
    if (D_star < .05) {
        // Dietrich 1982, eq. 8
        R_1 = log10(pow(D_star, 2) / 5832);
    } else if (D_star <= 5e9) {
        // Dietrich 1982, eq. 9
        R_1 = -3.76715 +
               1.92944 *     log10(D_star) -
               0.09815 * pow(log10(D_star), 2) -
               0.00575 * pow(log10(D_star), 3) +
               0.00056 * pow(log10(D_star), 4);
    } else {
        return NAN;  // flag failure
    }

    double R_2 = log10(1 - ((1 - CSF)/0.85))  // this coefficient takes shape into account
                - pow(1 - CSF, 2.3) * tanh(log10(D_star) - 4.6)
                + 0.3 * (0.5 - CSF) * pow(1 - CSF, 2) * (log10(D_star) - 4.6);

    return pow(10, R_1 + R_2);
}
