"""
advect on ECCO surface currents
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import timedelta
from opencl_driver_2D import openCL_advect
from plot_advection import plot_ocean_advection
from dask.diagnostics import ProgressBar


def test_ECCO():
    print('Opening ECCO current files...')
    U = xr.open_mfdataset('../forcing_data/ECCO/ECCO_interp/U_2015*.nc')
    V = xr.open_mfdataset('../forcing_data/ECCO/ECCO_interp/V_2015*.nc')
    currents = xr.merge((U, V)).sel(depth=0, method='nearest')

    print(f'Loading surface currents into RAM ({currents.nbytes/1e6:.0f} MB)...')
    with ProgressBar():
        currents.load()

    # create a land mask, then replace currents on land with 0 (easy method to get beaching)
    land = currents.U.isel(time=0).isnull()
    currents = currents.fillna(value=0)

    # initialize particles
    [X, Y] = np.meshgrid(currents.lon, currents.lat)
    ocean_points = np.array([X[~land], Y[~land]]).T
    num_particles = 5000
    p0 = pd.DataFrame(data=[ocean_points[i] for i in np.random.randint(0, len(ocean_points), size=num_particles)],
                      columns=['lon', 'lat'])

    # initialize advection parameters
    dt = timedelta(hours=1)
    time = pd.date_range(start='2015-01-01T12', end='2016-01-01T12', freq=dt, closed='left')
    save_every = 24

    print('Performing advection calculation...')
    P, buf_time, kernel_time = openCL_advect(field=currents, p0=p0, advect_time=time,
                                             save_every=save_every, platform_and_device=(0, 2), # change this to None for interactive device selection
                                             verbose=True)

    return P


if __name__ == '__main__':
    P = test_ECCO()
    plot_ocean_advection(P)
