"""
advect on ECCO surface currents
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import timedelta
from drivers.opencl_driver_2D import openCL_advect
from kernel_wrappers.Kernel2D import AdvectionScheme
from plotting.plot_advection import plot_ocean_advection


def test_ECCO():
    print('Opening ECCO current files...')
    U = xr.open_mfdataset('../forcing_data/ECCO/ECCO_interp/U_2015*.nc')
    V = xr.open_mfdataset('../forcing_data/ECCO/ECCO_interp/V_2015*.nc')
    currents = xr.merge((U, V)).sel(depth=0, method='nearest')

    # create a land mask
    land = currents.U.isel(time=0).isnull()

    # initialize particles
    [X, Y] = np.meshgrid(currents.lon, currents.lat)
    ocean_points = np.array([X[~land], Y[~land]]).T
    num_particles = 5000
    p0 = pd.DataFrame(data=[ocean_points[i] for i in np.random.randint(0, len(ocean_points), size=num_particles)],
                      columns=['lon', 'lat'])
    p0['release_date'] = np.concatenate((np.full(num_particles//2, np.datetime64('2015-01-01T12')),
                                        np.full(num_particles//2 + num_particles % 2, np.datetime64('2015-06-01'))))

    # initialize advection parameters
    dt = timedelta(hours=1)
    time = pd.date_range(start='2015-01-01T12', end='2016-01-01T12', freq=dt, closed='left')
    save_every = 24

    print('Performing advection calculation...')
    P, buf_time, kernel_time = openCL_advect(field=currents, p0=p0, advect_time=time, save_every=save_every,
                                             advection_scheme=AdvectionScheme.eulerian,
                                             platform_and_device=(0, 2), # change this to None for interactive device selection
                                             verbose=True)

    return P, (buf_time, kernel_time)


if __name__ == '__main__':
    P, (buf_time, kernel_time) = test_ECCO()
    plot_ocean_advection(P)
