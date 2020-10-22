#include "kernel_helpers.cl"

/* headers */
vector eulerian_displacement(particle p, unsigned int x_idx, unsigned int y_idx, unsigned int t_idx,
                            __global float* field_U, __global float* field_V,
                            __global double* field_x, unsigned int x_len,
                            __global double* field_y, unsigned int y_len,
                            unsigned int t_len, double dt);
vector taylor2_displacement(particle p, unsigned int x_idx, unsigned int y_idx, unsigned int t_idx,
                            __global float* field_U, __global float* field_V,
                            __global double* field_x, unsigned int x_len,
                            __global double* field_y, unsigned int y_len,
                            unsigned int t_len, double dt);

vector eulerian_displacement(particle p, unsigned int x_idx, unsigned int y_idx, unsigned int t_idx,
                            __global float* field_U, __global float* field_V,
                            __global double* field_x, unsigned int x_len,
                            __global double* field_y, unsigned int y_len,
                            unsigned int t_len, double dt) {
    // find U and V nearest to particle position
    float u = index_vector_field(field_U, x_len, y_len, x_idx, y_idx, t_idx);
    float v = index_vector_field(field_V, x_len, y_len, x_idx, y_idx, t_idx);

    //////////// advect particle using euler forward advection scheme
    // meters displacement
    vector displacement_meters;
    displacement_meters.x = u * dt;
    displacement_meters.y = v * dt;
    return displacement_meters;
}

vector taylor2_displacement(particle p, unsigned int x_idx, unsigned int y_idx, unsigned int t_idx,
                            __global float* field_U, __global float* field_V,
                            __global double* field_x, unsigned int x_len,
                            __global double* field_y, unsigned int y_len,
                            unsigned int t_len, double dt) {
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
    vector displacement_meters;
    displacement_meters.x = (u_ + (uy*v_ - vy*u_) * dt/2) * dt / ((1 - ux*dt/2) * (1 - vy*dt/2) - (uy*vx * pow(dt, 2)) / 4);
    displacement_meters.y = (v_ + (vx*u_ - ux*v_) * dt/2) * dt / ((1 - uy*dt/2) * (1 - vx*dt/2) - (ux*vy * pow(dt, 2)) / 4);

    return displacement_meters;
}
