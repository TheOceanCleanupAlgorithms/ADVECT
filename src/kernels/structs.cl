#ifndef STRUCTS
#define STRUCTS
/* definitions of useful structures for grouping related entities */
typedef struct dataset {
    double test;
} dataset;

typedef struct particle {
    int id;
    double x;
    double y;
    double t;
} particle;

typedef struct vector {
    double x;
    double y;
} vector;

#endif
