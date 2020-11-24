#include "vector.cl"

__kernel void resolve_and_sort_test(
    const double x,
    const double y,
    const double z,
    __global double *x_out,
    __global double *y_out,
    __global double *z_out) {
    vector v = {.x = x, .y = y, .z = z};

    vector sorted_resolution[3];
    resolve_and_sort(v, sorted_resolution);

    for (int i=0; i<3; i++) {
        x_out[i] = sorted_resolution[i].x;
        y_out[i] = sorted_resolution[i].y;
        z_out[i] = sorted_resolution[i].z;
    }
}
