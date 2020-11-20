#include "vector.h"

void * swap(void * arr, size_t i, size_t j);

vector add(vector a, vector b) {
    vector res = {.x = a.x + b.x,
                  .y = a.y + b.y,
                  .z = a.z + b.z};
    return res;
}


vector * resolve_and_sort(vector v) {
    /* Split v into 3 vectors, one for each dimension.  Return these sorted by magnitude. */
    vector v_x = {.x = v.x, .y = 0,   .z = 0  };
    vector v_y = {.x = 0,   .y = v.y, .z = 0  };
    vector v_z = {.x = 0,   .y = 0,   .z = v.z};
    vector sorted[3];

    // for sorting only 3 values, this is most efficient I think, and avoids writing a swap function.
    if (v.x < v.y) {
        if (v.x < v.z) {
            sorted[0] = v_x;
            if (v.y < v.z) {
                sorted[1] = v_y;
                sorted[2] = v_z;
            } else {
                sorted[1] = v_z;
                sorted[2] = v_y;
            }
        } else {
            sorted[0] = v_z;
            sorted[1] = v_x;
            sorted[2] = v_y;
        }
    } else {
        if (v.x > v.z) {
            sorted[2] = v_x;
            if (v.y < v.z) {
                sorted[0] = v_y;
                sorted[1] = v_z;
            } else {
                sorted[0] = v_z;
                sorted[1] = v_y;
            }
        } else {
            sorted[0] = v_y;
            sorted[1] = v_x;
            sorted[2] = v_z;
        }
    }
    return sorted;
}
