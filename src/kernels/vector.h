#ifndef VECTOR
#define VECTOR

typedef struct vector {
    double x;
    double y;
    double z;
} vector;

vector add(vector a, vector b);

#endif // VECTOR
