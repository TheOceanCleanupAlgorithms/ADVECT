#include "vertical_profile.h"

double sample_profile(vertical_profile profile, double z) {
    /* gets the value in profile at z by linear interpolation
     * if z outside profile domain, returns nearest neighbor
     */

     if (z <= profile.z[0]) {
        return profile.var[0];
     } else if (z >= profile.z[profile.len - 1]) {
        return profile.var[profile.len - 1];
     }

     // find two nearest neighbors, depth and variable value
     // remember, profile.z guaranteed ascending, and at this point guaranteed profile.z[0] < z < profile.z[profile.len - 1]
     double z0, var0, z1, var1;
     for (unsigned int i=1; i < profile.len; i++) {
        if (z < profile.z[i]) {
            z0 = profile.z[i-1];
            var0 = profile.var[i-1];
            z1 = profile.z[i];
            var1 = profile.var[i];
            break;
        }
    }

    // linear interpolation of profile.var at z between z0 and z1
    return var0 + (z - z0) * (var1 - var0) / (z1 - z0);
}
