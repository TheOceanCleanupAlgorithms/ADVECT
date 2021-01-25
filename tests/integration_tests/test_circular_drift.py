"""
The purpose of this test is to ensure that the second-order taylor kernel advects particles along circular streamlines.
It should also show that the Eulerian kernel fails to do this.
We will use a small latitude/longitude scale in order to approximate cartesian behavior.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pyopencl as cl
from datetime import timedelta
from io_tools.open_vectorfiles import empty_2D_vectorfield
from kernel_wrappers.Kernel3D import AdvectionScheme, Kernel3D


def compare_alg_drift(initial_radius: float, plot=False):
    nx = 20
    lon = np.linspace(-.02, .02, nx*2)  # .01 degrees at equator ~= 1 km
    lat = np.linspace(-.01, .01, nx)

    LON, LAT = np.meshgrid(lon, lat)
    mag = np.sqrt(LON**2 + LAT**2)
    U = np.divide(-LAT, mag, out=np.zeros_like(LON), where=mag != 0)
    V = np.divide(LON, mag, out=np.zeros_like(LON), where=mag != 0)

    current = xr.Dataset(
        {
            "U": (["lat", "lon", "depth", "time"], U[:, :, np.newaxis, np.newaxis]),
            "V": (["lat", "lon", "depth", "time"], V[:, :, np.newaxis, np.newaxis]),
            "W": (["lat", "lon", "depth", "time"], np.zeros((*U.shape, 1, 1))),
        },
        coords={
            "lon": lon,
            "lat": lat,
            "depth": [0],
            "time": [np.datetime64('2000-01-01')],
        },
    )

    p0 = pd.DataFrame(
        {'p_id': [0], 'lon': [0], 'lat': [initial_radius], 'depth': [0],
         'radius': [.001], 'density': [1025], 'corey_shape_factor': [1],
         'exit_code': [0]}
    )
    eddy_diffusivity = xr.Dataset(
        {"horizontal_diffusivity": ("z_hd", ([0])),
         "vertical_diffusivity": ("z_vd", ([0]))},  # neutral buoyancy
        coords={"z_hd": [0], "z_vd": [0]}
    )
    density_profile = xr.Dataset(
        {"seawater_density": ("z_sd", (p0.density.values))},  # neutral buoyancy
        coords={"z_sd": [0]}
    )

    dt = timedelta(seconds=30)
    time = pd.date_range(start='2000-01-01', end='2000-01-01T6:00:00', freq=dt)
    p0['release_date'] = time[0]
    p0 = xr.Dataset(p0.set_index('p_id'))
    save_every = 1
    wind = empty_2D_vectorfield()

    euler = Kernel3D(current=current, wind=wind, p0=p0,
                     advect_time=time, save_every=save_every,
                     advection_scheme=AdvectionScheme.eulerian,
                     eddy_diffusivity=eddy_diffusivity,
                     density_profile=density_profile,
                     max_wave_height=0, wave_mixing_depth_factor=0,
                     windage_multiplier=None, context=cl.create_some_context()).execute().squeeze()

    taylor = Kernel3D(current=current, p0=p0, wind=wind,
                      advect_time=time, save_every=save_every,
                      advection_scheme=AdvectionScheme.taylor2,
                      eddy_diffusivity=eddy_diffusivity,
                      density_profile=density_profile,
                      max_wave_height=0, wave_mixing_depth_factor=0,
                      windage_multiplier=None, context=cl.create_some_context()).execute().squeeze()

    if plot:
        plt.figure(figsize=(8, 4))
        ax = plt.axes()
        ax.quiver(current.lon, current.lat, current.U.squeeze(), current.V.squeeze())
        ax.plot(p0.lon, p0.lat, 'go')

        for name, P in {'euler': euler, 'taylor': taylor}.items():
            ax.plot(P.lon, P.lat, '.-', label=name, linewidth=2)
            ax.plot(P.isel(time=-1).lon, P.isel(time=-1).lat, 'rs')
        plt.legend()
        plt.title('drift comparison in circular field')
        plt.show()

    return euler, taylor


def test_circular_drift():
    initial_radius = .005
    euler, taylor = compare_alg_drift(initial_radius=initial_radius)

    # taylor stays within 5% of radius on average
    np.testing.assert_allclose(np.sqrt(taylor.lon ** 2 + taylor.lat ** 2).mean(),
                               initial_radius, rtol=.05)

    # euler spirals out wildly
    assert np.sqrt(euler.lon ** 2 + euler.lat ** 2)[-1] > initial_radius * 1.5


if __name__ == '__main__':
    euler, taylor = compare_alg_drift(initial_radius=.005, plot=True)
