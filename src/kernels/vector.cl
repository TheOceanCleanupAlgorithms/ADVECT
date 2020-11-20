#include "vector.h"

void * swap(void * arr, size_t i, size_t j);

vector add(vector a, vector b) {
    vector res = {.x = a.x + b.x,
                  .y = a.y + b.y,
                  .z = a.z + b.z};
    return res;
}


void resolve_and_sort(vector v, vector result[3]) {
    /* Split v into 3 vectors, one for each dimension.  Return these sorted by magnitude.
       Return type: vector[3] */
    vector v_x = {.x = v.x, .y = 0,   .z = 0  };
    vector v_y = {.x = 0,   .y = v.y, .z = 0  };
    vector v_z = {.x = 0,   .y = 0,   .z = v.z};

    double mag_x = fabs(v.x);
    double mag_y = fabs(v.y);
    double mag_z = fabs(v.z);

    // for sorting only 3 values, this is most efficient I think, and avoids writing a swap function.
    if (mag_x < mag_y) {
        if (mag_x < mag_z) {
            result[0] = v_x;
            if (mag_y < mag_z) {
                result[1] = v_y;
                result[2] = v_z;
            } else {
                result[1] = v_z;
                result[2] = v_y;
            }
        } else {
            result[0] = v_z;
            result[1] = v_x;
            result[2] = v_y;
        }
    } else {
        if (mag_x > mag_z) {
            result[2] = v_x;
            if (mag_y < mag_z) {
                result[0] = v_y;
                result[1] = v_z;
            } else {
                result[0] = v_z;
                result[1] = v_y;
            }
        } else {
            result[0] = v_y;
            result[1] = v_x;
            result[2] = v_z;
        }
    }
}
