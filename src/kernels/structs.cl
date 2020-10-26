#ifndef STRUCTS
#define STRUCTS
/* definitions of useful structures for grouping related entities */
typedef struct field2d {
    __global const double *x, *y, *t;
    const unsigned int x_len, y_len, t_len;
    const double x_spacing, y_spacing, t_spacing;
    __global const float *U, *V;
} field2d;

typedef struct grid_point {
    unsigned int x_idx;
    unsigned int y_idx;
    unsigned int t_idx;
} grid_point;

typedef struct particle {
    const int id;
    double x;
    double y;
    double t;
} particle;

typedef struct vector {
    double x;
    double y;
} vector;

typedef struct random_state { // for the xor random number generator
  unsigned int a;  // 32 bits
} random_state;

#endif
