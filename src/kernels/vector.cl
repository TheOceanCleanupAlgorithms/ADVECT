#include "vector.h"

vector add(vector a, vector b) {
    vector res = {.x = a.x + b.x, .y = a.y + b.y};
    return res;
}
