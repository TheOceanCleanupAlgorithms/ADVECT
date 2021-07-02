#ifndef VECTOR
#define VECTOR

typedef struct vector {
    double x;
    double y;
    double z;
} vector;

vector add(vector a, vector b);
vector mul(vector v, double c);
void resolve_and_sort(vector v, vector result[3]);
double magnitude(vector v);

#endif // VECTOR
