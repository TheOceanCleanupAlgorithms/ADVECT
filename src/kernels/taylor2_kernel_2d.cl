unsigned int find_nearest_neighbor_idx(double value, __global double* arr, const unsigned int arr_len, const double spacing);
float index_vector_field(float* field, unsigned int x_len, unsigned int y_len,
                         unsigned int x_idx, unsigned int y_idx, unsigned int t_idx);
float meters_to_degrees_lon(float dx_meters, float y);
float meters_to_degrees_lat(float dy_meters, float y);
float degrees_lon_to_meters(float dx, float y);
float degrees_lat_to_meters(float dy, float y);

__kernel void advect(
    __global double* field_x,    // lon, Deg E (-180 to 180), uniform spacing
    const unsigned int x_len,   // <= UINT_MAX + 1
    __global double* field_y,    // lat, Deg N (-90 to 90), uniform spacing
    const unsigned int y_len,   // <= UINT_MAX + 1
    __global double* field_t,     // time, seconds since epoch, uniform spacing
    const unsigned int t_len,   // <= UINT_MAX + 1
    __global float* field_U,    // m / s, 32 bit to save space
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
    const double x_spacing = field_x[1]-field_x[0];
    const double y_spacing = field_y[1]-field_y[0];
    const double t_spacing = field_t[1]-field_t[0];

    // loop timesteps
    double x = x0[p_id];
    double y = y0[p_id];
    double t = t0[p_id];
    for (unsigned int timestep=0; timestep<ntimesteps; timestep++) {

        // find nearest neighbors in grid
        unsigned int x_idx = find_nearest_neighbor_idx(x, field_x, x_len, x_spacing);
        unsigned int y_idx = find_nearest_neighbor_idx(y, field_y, y_len, y_spacing);
        unsigned int t_idx = find_nearest_neighbor_idx(t, field_t, t_len, t_spacing);

        //////////// advect particle using second-order taylor approx advection scheme (Black and Gay, 199?)
        // meters displacement

        float u = index_vector_field(field_U, x_len, y_len, x_idx, y_idx, t_idx);  // u at the particle
        float v = index_vector_field(field_V, x_len, y_len, x_idx, y_idx, t_idx);  // v at the particle
        float u_w = index_vector_field(field_U, x_len, y_len, (x_idx - 1) % x_len, y_idx, t_idx);  // u one grid cell left
        float u_e = index_vector_field(field_U, x_len, y_len, (x_idx + 1) % x_len, y_idx, t_idx);  // u one grid cell right
        float u_s = index_vector_field(field_U, x_len, y_len, x_idx, max(y_idx - 1, 0), t_idx);  // u one grid cell down
        float u_n = index_vector_field(field_U, x_len, y_len, x_idx, min(y_idx + 1, y_len - 1), t_idx);  // u one grid cell up
        float v_w = index_vector_field(field_V, x_len, y_len, (x_idx - 1) % x_len, y_idx, t_idx);  // v one grid cell left
        float v_e = index_vector_field(field_V, x_len, y_len, (x_idx + 1) % x_len, y_idx, t_idx);  // v one grid cell right
        float v_s = index_vector_field(field_V, x_len, y_len, x_idx, max(y_idx - 1, 0), t_idx);  // v one grid cell down
        float v_n = index_vector_field(field_V, x_len, y_len, x_idx, min(y_idx + 1, y_len - 1), t_idx);  // v one grid cell up
        float u_dt = index_vector_field(field_U, x_len, y_len, x_idx, y_idx, min(t_idx + 1, t_len - 1));  // u at particle position one index in future
        float v_dt = index_vector_field(field_V, x_len, y_len, x_idx, y_idx, min(t_idx + 1, t_len - 1));  // v at particle position one index in future

        // grid spacing at particle in x direction (m)
        if (x_idx == 0) {
            float dx = field_x[x_idx + 1] - field_x[x_idx];
        } else {
            float dy = field_x[x_idx] - field_x[x_idx - 1];
        }
        float dx_m = degrees_lon_to_meters(dx, y);

        // grid spacing at particle in y direction (m)
        if (y_idx == 0) {
            float dy = field_y[y_idx + 1] - field_y[y_idx];
        } else {
            float dy = field_y[y_idx] - field_y[y_idx - 1];
        }
        float dy_m = degrees_lat_to_meters(dy, y);

        // Calculate horizontal gradients
        float ux = (u_w - u_e) / (2*dx_m);
        float uy = (u_s - u_n) / (2*dy_m);
        float vx = (v_w - v_e) / (2*dx_m);
        float vy = (v_s - v_n) / (2*dy_m);

        // Calculate time gradients
        float ut = (u_dt - u) / dt;
        float vt = (v_dt - v) / dt;

        // simplifying term
        float u_ = u + (dt*ut)/2;  // these units don't even make sense...
        float v_ = v + (dt*vt)/2;

        float u_taylor = ( u_ + ( uy.*v_ - vy.*u_ )*dt/2 ) ./ ( (1-ux*dt/2).*(1-vy*dt/2) - (uy.*vx*dt^2)/4 ) ;
        float v_taylor = ( v_ + ( vx.*u_ - ux.*v_ )*dt/2 ) ./ ( (1-uy*dt/2).*(1-vx*dt/2) - (ux.*vy*dt^2)/4 ) ;

        float x_disp_meters = u_taylor * dt;
        float y_disp_meters = v_taylor * dt;

        float dx_deg = meters_to_degrees_lon(x_disp_meters, y)
        float dy_deg = meters_to_degrees_lat(y_disp_meters, y)

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

unsigned int find_nearest_neighbor_idx(double value, __global double* arr, const unsigned int arr_len, const double spacing) {
    // assumption: arr is sorted with uniform spacing.  Actually works on ascending or descending sorted arr.
    // also, we must have arr_len - 1 <= UINT_MAX for the cast of the clamp result to behave properly.  Can't raise errors
    // inside a kernel so we must perform the check in the host code.
    return (unsigned int) clamp(round((value - arr[0])/spacing), (double) (0.0), (double) (arr_len-1));
}

float index_vector_field(float* field, unsigned int x_len, unsigned int y_len,
                         unsigned int x_idx, unsigned int y_idx, unsigned int t_idx) {
    /*
    field: the vector field from which to retrieve a value.  Dimensions (time, x, y)
    [dim]_len: the length of the [dim] dimension of 'field'
    [dim]_idx: the index along the [dim] dimension
    assumption: [dim]_idx args will be in [0, [dim]_len - 1]
    */

    return field[(t_idx*x_len + x_idx)*y_len + y_idx];


// convert meters displacement to lat/lon and back(Reference: American Practical Navigator, Vol II, 1975 Edition, p 5)
float meters_to_degrees_lon(float dx_meters, float y) {
    float rlat = y * M_PI/180;
    return dx_meters / (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

float meters_to_degrees_lat(float dy_meters, float y) {
    float rlat = y * M_PI/180;
    return dy_meters / (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}

float degrees_lon_to_meters(float dx, float y) {
    float rlat = y * M_PI/180;
    return dx * (111415.13 * cos(rlat) - 94.55 * cos(3 * rlat));
}

float degrees_lat_to_meters(float dy, float y) {
    float rlat = y * M_PI/180;
    return dy * (111132.09 - 556.05 * cos(2 * rlat) + 1.2 * cos(4 * rlat));
}
