"""
The purpose of this test is to ensure that the second-order taylor kernel advects particles along circular streamlines.
It should also show that the Eulerian kernel fails to do this.
We will use a small latitude/longitude scale in order to approximate cartesian behavior.
"""
from datetime import timedelta

import numpy as np
import xarray as xr
import pandas as pd

from drivers.opencl_driver_2D import openCL_advect
from kernel_wrappers.EulerianKernel2D import EulerianKernel2D
from kernel_wrappers.Taylor2Kernel2D import Taylor2Kernel2D
import matplotlib.pyplot as plt

def compare_alg_drift():
    nx = 20
    lon = np.linspace(-.02, .02, nx*2)  # .01 degrees at equator ~= 1 km
    lat = np.linspace(-.01, .01, nx)

    LON, LAT = np.meshgrid(lon, lat)
    mag = np.sqrt(LON**2 + LAT**2)
    U = np.divide(-LAT, mag, out=np.zeros_like(LON), where=mag != 0)
    V = np.divide(LON, mag, out=np.zeros_like(LON), where=mag != 0)

    field = xr.Dataset(
        {
            "U": (["lat", "lon", "time"], U[:, :, np.newaxis]),
            "V": (["lat", "lon", "time"], V[:, :, np.newaxis]),
        },
        coords={
            "lon": lon,
            "lat": lat,
            "time": [np.datetime64('2000-01-01')],
        },
    )

    p0 = pd.DataFrame({'lon': [0], 'lat': [.001]})
    num_particles = 1
    dt = timedelta(seconds=20)
    time = pd.date_range(start='2000-01-01', end='2000-01-01T18:00:00', freq=dt)
    save_every = 1

    euler, _, _ = openCL_advect(field=field, p0=p0, advect_time=time, save_every=save_every, kernel_class=EulerianKernel2D,
                                platform_and_device=(0, 0), verbose=True)

    taylor, _, _ = openCL_advect(field=field, p0=p0, advect_time=time, save_every=save_every,  kernel_class=Taylor2Kernel2D,
                                 platform_and_device=(0, 0), verbose=True)

    plt.figure()
    ax = plt.axes()
    ax.quiver(field.lon, field.lat, field.U.isel(time=0), field.V.isel(time=0))
    ax.plot(p0.lon, p0.lat, 'go')

    for name, P in {'euler': euler, 'taylor': taylor}.items():
        ax.plot(P.isel(p_id=0).lon, P.isel(p_id=0).lat, label=name)
        ax.plot(P.isel(p_id=0, time=-1).lon, P.isel(p_id=0, time=-1).lat, 'rs')
    plt.legend()
    plt.title('drift comparison in circular field')


if __name__ == '__main__':
    compare_alg_drift()
