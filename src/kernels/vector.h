#ifndef VECTOR
#define VECTOR

typedef struct vector {
    double x;
    double y;
    double z;
} vector;

vector add(vector a, vector b);
double magnitude(vector v);
vector * resolve_and_sort(vector v);

#endif // VECTOR
