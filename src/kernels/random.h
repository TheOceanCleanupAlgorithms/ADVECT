#ifndef RANDOM
#define RANDOM

typedef struct random_state {
  unsigned int a;  // 32 bits
} random_state;

double random(random_state *state);

#endif // RANDOM
