#include "diffusion.cl"
#include "vertical_profile.cl"
#include "random.cl"

__kernel void single_diffusion_step(
    __global const double *z,  // depth of diffusion, one per thread
    const double dt,
    __global const unsigned int *seed,  // seeds for each thread
    __global const double *horizontal_eddy_diffusivity_z,  // depth coordinates, m, positive up, sorted ascending
    __global const double *horizontal_eddy_diffusivity_values,    // m^2 s^-1
    const unsigned int horizontal_eddy_diffusivity_len,
    __global const double *vertical_eddy_diffusivity_z,  // depth coordinates, m, positive up, sorted ascending
    __global const double *vertical_eddy_diffusivity_values,    // m^2 s^-1
    const unsigned int vertical_eddy_diffusivity_len,
    __global double3 *out) {
    vertical_profile hdiff = {
        .values = horizontal_eddy_diffusivity_values,
        .z = horizontal_eddy_diffusivity_z,
        .len = horizontal_eddy_diffusivity_len};
    vertical_profile vdiff = {
        .values = vertical_eddy_diffusivity_values,
        .z = vertical_eddy_diffusivity_z,
        .len = vertical_eddy_diffusivity_len};
    random_state rstate = {.a = seed[get_global_id(0)]};

    vector result = eddy_diffusion_meters(z[get_global_id(0)], dt, &rstate, hdiff, vdiff);

    out[get_global_id(0)].xyz = (double3)(result.x, result.y, result.z);
}
