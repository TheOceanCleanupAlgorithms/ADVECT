#include "kernel_helpers.cl"

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
    const unsigned int out_timesteps = ntimesteps / save_every;

    // calculate spacing of grids
    const double x_spacing = field_x[1]-field_x[0];
    const double y_spacing = field_y[1]-field_y[0];
    const double t_spacing = field_t[1]-field_t[0];

    // loop timesteps
    particle p;
    p.id = get_global_id(0);  // id of particle = id of thread
    p.x = x0[p.id];
    p.y = y0[p.id];
    p.t = t0[p.id];
    for (unsigned int timestep=0; timestep<ntimesteps; timestep++) {
        // find nearest neighbors in grid
        unsigned int x_idx = find_nearest_neighbor_idx(p.x, field_x, x_len, x_spacing);
        unsigned int y_idx = find_nearest_neighbor_idx(p.y, field_y, y_len, y_spacing);
        unsigned int t_idx = find_nearest_neighbor_idx(p.t, field_t, t_len, t_spacing);

        // find U and V nearest to particle position
        float u = index_vector_field(field_U, x_len, y_len, x_idx, y_idx, t_idx);
        float v = index_vector_field(field_V, x_len, y_len, x_idx, y_idx, t_idx);

        //////////// advect particle using euler forward advection scheme
        // meters displacement
        float dx_meters = u * dt;
        float dy_meters = v * dt;

        float dx_deg = meters_to_degrees_lon(dx_meters, p.y);
        float dy_deg = meters_to_degrees_lat(dy_meters, p.y);

        p = update_position(p, dx_deg, dy_deg, dt);

        // save if necessary
        if ((timestep+1) % save_every == 0) {
            unsigned int out_idx = (timestep+1)/save_every - 1;
            write_p(p, X_out, Y_out, out_timesteps, out_idx);
        }
    }
}
