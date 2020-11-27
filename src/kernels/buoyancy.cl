#include "buoyancy.h"
#include "physical_constants.h"

double estimate_w_terminal(particle p);

particle update_w_terminal(particle p) {
    /* Update the terminal velocity on a particle.
       Uses p.w_terminal as a characteristic velocity for calculating drag coefficient
       Calling this function repeatedly will converge to the particle's true terminal velocity.
       Density and kinematic viscosity of seawater are considered constant.
       If particle does not yet posses terminal velocity state, set p.w_terminal = NAN.
    */

    if (DENSITY_SEAWATER == p.rho) {  // neutrally buoyant, this all doesn't apply.
        p.w_terminal = 0;
        return p;
    }

    if (isnan(p.w_terminal) || p.w_terminal == 0) {  // this indicates we need to guess an initial state
        p.w_terminal = estimate_w_terminal(p);
    }

    double V_p = 4/3 * PI * pow(p.r, 3);  // volume of particle
    double A_p = PI * pow(p.r, 2);  // cross sectional area

    double F_b = -(DENSITY_SEAWATER * V_p * ACC_GRAVITY);  // buoyancy force
    double F_g = p.rho * V_p * ACC_GRAVITY;  // force due to gravity
    // we know forces balance (because we assume terminal velocity) thus
    double F_d = -(F_b + F_g);  // drag force

    // drag coefficient
    if (isnan(p.w_terminal) || p.w_terminal == 0) {  // indicates we need to guess an initial state
        C_D = .5;
    } else {
        C_D = 2 * F_d / (DENSITY_SEAWATER * pow(p.w_terminal, 2) * A_p);
    }

    double terminal_velocity = sqrt((2 * ACC_GRAVITY * p.r) / (3 * C_d) * (p.rho - DENSITY_SEAWATER)/DENSITY_SEAWATER);

    p.w_terminal = settling_velocity;
}

double estimate_w_terminal(particle p) {
    return 0;
}