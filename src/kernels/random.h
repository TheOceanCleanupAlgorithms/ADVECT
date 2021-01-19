#ifndef RANDOM
#define RANDOM

typedef struct random_state {
  unsigned int a;  // 32 bits
} random_state;

double random_within_magnitude(double magnitude, random_state *rstate);
double random(random_state *rstate);
double random_in_range(double low, double high, random_state *rstate);
double random_normal(double mean, double std, random_state *rstate);

#endif // RANDOM
