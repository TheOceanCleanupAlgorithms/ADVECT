#include "diffusion.h"

double amplitude_of_diffusion(const double dt, unsigned int ndims, double diffusivity);

vector eddy_diffusion_meters(double z, const double dt, random_state *rstate,
                             vertical_profile horizontal_eddy_diffusivity_profile,
                             vertical_profile vertical_eddy_diffusivity_profile) {
    /* Use the vertical profiles of eddy diffusivity to generate a step of motion due to
       eddy diffusion at depth z.
     */
    double horizontal_eddy_diffusivity = sample_profile(horizontal_eddy_diffusivity_profile, z);
    double vertical_eddy_diffusivity = sample_profile(vertical_eddy_diffusivity_profile, z);
    vector diffusion = {.x = diffusion_step(horizontal_eddy_diffusivity, dt, rstate),
                        .y = diffusion_step(horizontal_eddy_diffusivity, dt, rstate),
                        .z = diffusion_step(vertical_eddy_diffusivity, dt, rstate)};
    return diffusion;
}


double diffusion_step(double diffusivity, const double dt, random_state *rstate) {
    /* Calculate a displacement due to one-dimensional turbulent diffusion.
       Diffusion is represented by a Wiener process, which is more or less a random walk in each dimension
       where the amplitude of each step is chosen from a normal distribution whose variance is proportional to the
       the timestep, as well as the eddy diffusivity.
       See "Stochastic Lagrangian Models of Turbelent Diffusion", Howard Rodean 1996, eq. 8.43.
     * dt: timestep in seconds
     * diffusivity: m^2 s^-1
     * return: displacement in meters
    */
    double dW = random_normal(0, sqrt(dt), rstate);  // Wiener step, eq. 4.3a
    return sqrt(2*diffusivity) * dW;  // simplified form of eq. 8.43, Rodean 1996
}
