unsigned int find_nearest_neighbor_idx(float value, __global float* arr, const unsigned int arr_len, const float spacing);

__kernel void advect(
    __global float* field_x,    // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int x_len,   // <= UINT_MAX
    __global float* field_y,    // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int y_len,   // <= UINT_MAX
    __global float* field_t,    // time, unix timestamp, uniform spacing
    const unsigned int t_len,   // <= UINT_MAX
    __global float* field_U,    // m / s
    __global float* field_V,    // m / s
    __global float* x0,         // lon, Deg E (-180 to 180)
    __global float* y0,         // lat, Deg N (-90 to 90)
    __global float* t0,         // unix timestamp
    const float dt,             // seconds
    const unsigned int ntimesteps,
    const unsigned int save_every,
    __global float* X_out,      // lon, Deg E (-180 to 180)
    __global float* Y_out)      // lat, Deg N (-90 to 90)
{
    int p_id = get_global_id(0);  // id of particle
    const unsigned int out_timesteps = ntimesteps / save_every;

    // calculate spacing of grids
    const float x_spacing = field_x[1]-field_x[0];
    const float y_spacing = field_y[1]-field_y[0];
    const float t_spacing = field_t[1]-field_t[0];

    // loop timesteps
    float x = x0[p_id];
    float y = y0[p_id];
    float t = t0[p_id];
    for (unsigned int timestep=0; timestep<ntimesteps; timestep++) {

        // find nearest neighbors in grid
        unsigned int x_idx = find_nearest_neighbor_idx(x, field_x, x_len, x_spacing);
        unsigned int y_idx = find_nearest_neighbor_idx(y, field_y, y_len, y_spacing);
        unsigned int t_idx = find_nearest_neighbor_idx(t, field_t, t_len, t_spacing);

        // find U and V nearest to particle position
        float u = field_U[(t_idx*x_len + x_idx)*y_len + y_idx];
        float v = field_V[(t_idx*x_len + x_idx)*y_len + y_idx];

        //////////// advect particle
        // meters displacement
        float dx_meters = u * dt;
        float dy_meters = v * dt;

        // convert meters displacement to lat/lon (Reference: American Practical Navigator, Vol II, 1975 Edition, p 5)
        float rlat = y * M_PI/180;
        float dx_deg = dx_meters / (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
        float dy_deg = dy_meters / (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));

        // update
        x = x + dx_deg;
        y = y + dy_deg;
        t = t + dt;
        // deal with advecting over the poles
        if (y > 90) {
            y = 180 - y;
            x = x + 180;
        } else if (y < -90) {
            y = -180 - y;
            x = x + 180;
        }
        // keep longitude representation within [-180, 180)
        // builtin fmod(a, b) is a - b*trunc(a/b), which behaves incorrectly for negative numbers.
        //            so we use  a - b*floor(a/b) instead
        x = ((x+180) - 360*floor((x+180)/360)) - 180;

        // save if necessary
        if ((timestep+1) % save_every == 0) {
            unsigned int out_idx = (timestep+1)/save_every - 1;
            X_out[p_id*out_timesteps + out_idx] = x;
            Y_out[p_id*out_timesteps + out_idx] = y;
        }
    }
}

unsigned int find_nearest_neighbor_idx(float value, __global float* arr, const unsigned int arr_len, const float spacing) {
    // assumption: arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
    // also, we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.  Can't raise errors
    // inside a kernel so we must perform the check in the host code.
    return (unsigned int) clamp(round((value - arr[0])/spacing), (float) (0.0), (float) (arr_len-1));
}
