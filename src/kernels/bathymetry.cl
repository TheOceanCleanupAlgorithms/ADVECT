#include "bathymetry.h"
#include "fields.h"

bool location_is_in_ocean(vector location, bathymetry bathy) {
    /* whene're you are below bathymetry,
       you sure as heck can bet this ain't the sea.*/
    unsigned int x_idx = find_nearest_neighbor_idx(location.x, bathy.x, bathy.x_len, bathy.x_spacing);
    unsigned int y_idx = find_nearest_neighbor_idx(location.y, bathy.y, bathy.y_len, bathy.y_spacing);
    float land_elevation = bathy.Z[y_idx*bathy.x_len + x_idx];

    return (land_elevation < location.z && location.z <= 0);  // aka, location is above ground but below sea-level.
                      // note that by this definition, places like, say, death valley, are considered the ocean.
                      // unclear how to get around this.  In theory particles could clip from ocean to a below-sea-level
                      // location on land, and they would never register as leaving the ocean.
}
