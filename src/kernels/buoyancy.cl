#include "buoyancy.h"
#include "physical_constants.h"

vector buoyancy_transport(particle p, double dt) {
    /* calculate displacement of particle due to density differential between particle and seawater.
       Assumes particle is always at terminal velocity appropriate for its buoyancy/radius.
       Terminal velocity given by method in Dietrich 1982
       Density and kinematic viscosity of seawater are considered constant.

       If return vector's z component is NAN, this indicates failure due to particle radius being too large.
    */
    vector displacement_meters = {.x = 0, .y = 0, .z = 0};

    double D_star = fabs((p.rho - DENSITY_SEAWATER) * ACC_GRAVITY * pow(2*p.r, 3) /
                    (DENSITY_SEAWATER * pow(KINEMATIC_VISCOSITY_SEAWATER, 2)));

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
        displacement_meters.z = NAN;  // flag failure
        return displacement_meters;
    }

    // Dietrich 1982, eq. 5 rearranged
    double settling_velocity = cbrt((W_star * (DENSITY_SEAWATER - p.rho) * ACC_GRAVITY * KINEMATIC_VISCOSITY_SEAWATER) / DENSITY_SEAWATER);
    displacement_meters.z = -settling_velocity * dt;
    return displacement_meters;
}
