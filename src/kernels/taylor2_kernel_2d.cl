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
    const double dt,             // seconds
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

        // find nearest neighbor grid cell
        unsigned int x_idx = find_nearest_neighbor_idx(p.x, field_x, x_len, x_spacing);
        unsigned int y_idx = find_nearest_neighbor_idx(p.y, field_y, y_len, y_spacing);
        unsigned int t_idx = find_nearest_neighbor_idx(p.t, field_t, t_len, t_spacing);

        // find adjacent cells
        unsigned int x_idx_w = (x_idx + 1) % x_len;
        unsigned int x_idx_e = (x_idx - 1) % x_len;
        unsigned int y_idx_s = max(y_idx - 1, 0u);
        unsigned int y_idx_n = min(y_idx + 1, y_len - 1);
        unsigned int t_idx_dt = min(t_idx + 1, t_len - 1);

        // extract values from nearest neighbor + adjacent cells
        float u = index_vector_field(field_U, x_len, y_len, x_idx, y_idx, t_idx);       // at the particle
        float v = index_vector_field(field_V, x_len, y_len, x_idx, y_idx, t_idx);       //
        float u_w = index_vector_field(field_U, x_len, y_len, x_idx_w, y_idx, t_idx);   // one grid cell left
        float v_w = index_vector_field(field_V, x_len, y_len, x_idx_w, y_idx, t_idx);   //
        float u_e = index_vector_field(field_U, x_len, y_len, x_idx_e, y_idx, t_idx);   // one grid cell right
        float v_e = index_vector_field(field_V, x_len, y_len, x_idx_e, y_idx, t_idx);   //
        float u_s = index_vector_field(field_U, x_len, y_len, x_idx, y_idx_s, t_idx);   // one grid cell down
        float v_s = index_vector_field(field_V, x_len, y_len, x_idx, y_idx_s, t_idx);   //
        float u_n = index_vector_field(field_U, x_len, y_len, x_idx, y_idx_n, t_idx);   // one grid cell up
        float v_n = index_vector_field(field_V, x_len, y_len, x_idx, y_idx_n, t_idx);   //
        float u_dt = index_vector_field(field_U, x_len, y_len, x_idx, y_idx, t_idx_dt);  // one grid cell in future
        float v_dt = index_vector_field(field_V, x_len, y_len, x_idx, y_idx, t_idx_dt);  //

        // grid spacing at particle in x direction (m)
        double dx;
        if (x_idx == 0) {
            dx = field_x[x_idx + 1] - field_x[x_idx];
        } else {
            dx = field_x[x_idx] - field_x[x_idx - 1];
        }
        double dx_m = degrees_lon_to_meters(dx, p.y);

        // grid spacing at particle in y direction (m)
        double dy;
        if (y_idx == 0) {
            dy = field_y[y_idx + 1] - field_y[y_idx];
        } else {
            dy = field_y[y_idx] - field_y[y_idx - 1];
        }
        double dy_m = degrees_lat_to_meters(dy, p.y);

        // Calculate horizontal gradients
        double ux = (u_w - u_e) / (2*dx_m);
        double vx = (v_w - v_e) / (2*dx_m);
        double uy = (u_s - u_n) / (2*dy_m);
        double vy = (v_s - v_n) / (2*dy_m);

        // Calculate time gradients
        double ut = (u_dt - u) / dt;
        double vt = (v_dt - v) / dt;

        // simplifying term
        double u_ = u + (dt*ut)/2;
        double v_ = v + (dt*vt)/2;

        //////////// advect particle using second-order taylor approx advection scheme (Black and Gay, 1990, eq. 12/13)
        double x_disp_meters = (u_ + (uy*v_ - vy*u_) * dt/2) * dt / ((1 - ux*dt/2) * (1 - vy*dt/2) - (uy*vx * pow(dt, 2)) / 4);
        double y_disp_meters = (v_ + (vx*u_ - ux*v_) * dt/2) * dt / ((1 - uy*dt/2) * (1 - vx*dt/2) - (ux*vy * pow(dt, 2)) / 4);

        double dx_deg = meters_to_degrees_lon(x_disp_meters, p.y);
        double dy_deg = meters_to_degrees_lat(y_disp_meters, p.y);

        p = update_position(p, dx_deg, dy_deg, dt);

        // save if necessary
        if ((timestep+1) % save_every == 0) {
            unsigned int out_idx = (timestep+1)/save_every - 1;
            write_p(p, X_out, Y_out, out_timesteps, out_idx);
        }
    }
}
