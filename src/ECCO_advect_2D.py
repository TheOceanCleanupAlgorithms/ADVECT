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


EDDY_DIFFUSIVITY = 0  # m^2 / s
''' Sylvia Cole et al 2015: diffusivity calculated at a 300km eddy scale, global average in top 1000m, Argo float data.
  This paper shows 2 orders of magnitude variation regionally, not resolving regional differences is a big error source.
  Additionally, the assumption here is that 300km eddies are not resolved by the velocity field itself.  If they are,
  we're doubling up on the eddy transport.  For reference, to resolve 300km eddies, the grid scale probably needs to be
  on order 30km, which at the equator would be ~1/3 degree.
'''


def test_ECCO():
    print('Opening ECCO current files...')
    U = xr.open_mfdataset('../forcing_data/ECCO/ECCO_interp/U_2015*.nc')
    V = xr.open_mfdataset('../forcing_data/ECCO/ECCO_interp/V_2015*.nc')
    currents = xr.merge((U, V)).sel(depth=0, method='nearest')

    # create a land mask, then replace currents on land with 0 (easy method to get beaching)
    land = currents.U.isel(time=0).isnull()
    currents = currents.fillna(value=0)

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
                                             advection_scheme=AdvectionScheme.taylor2, eddy_diffusivity=EDDY_DIFFUSIVITY,
                                             platform_and_device=(0, 2), # change this to None for interactive device selection
                                             verbose=True)

    return P, (buf_time, kernel_time)


if __name__ == '__main__':
    P, (buf_time, kernel_time) = test_ECCO()
    plot_ocean_advection(P)
