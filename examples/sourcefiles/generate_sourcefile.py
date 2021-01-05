import click
import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path


@click.command()
@click.option('-n', 'num_particles', required=False, default=5000)
@click.option('-rho', 'density', required=False, default=1025)
@click.option('-r', 'radius', required=False, default=.001)
@click.option('-z', 'depth', required=False, default=0.0)
@click.option('-o', 'out_name', required=False, type=click.Path(exists=False, dir_okay=False),
              default='neutral.nc')
def generate_sourcefile(
    num_particles: int,
    density: float,
    radius: float,
    depth: float,
    out_name: str,
):
    """
    :param num_particles: number of particles to generate
    :param density: density of particles, kg m^-3
    :param radius: radius of particles, m
    :param depth: starting depth of particles, m (positive up)
    :param out_name: sourcefile name
    :return:
    """
    # create a land mask
    land = xr.open_dataarray(Path(__file__).parent / 'land_mask.nc')

    # initialize particles
    [X, Y] = np.meshgrid(land.lon, land.lat)
    ocean_points = np.array([X[~land], Y[~land]]).T
    p0 = pd.DataFrame(data=[ocean_points[i] for i in np.random.randint(0, len(ocean_points), size=num_particles)],
                      columns=['lon', 'lat'])
    p0['release_date'] = np.datetime64('2015-01-01T12')
    # p0['release_date'] = np.concatenate((np.full(num_particles//2, np.datetime64('2015-01-01T12')),
    #                                      np.full(num_particles//2 + num_particles % 2, np.datetime64('2015-06-01'))))
    p0['depth'] = depth

    p0['radius'] = radius
    p0['density'] = density

    p0['p_id'] = np.arange(num_particles)
    ds = xr.Dataset(p0.set_index('p_id'))

    ds.attrs["title"] = f"Sourcefile for ADVECTOR"
    ds.attrs["institution"] = "The Ocean Cleanup"

    out_path = Path(__file__).parent / out_name
    ds.to_netcdf(out_path)


if __name__ == '__main__':
    generate_sourcefile()
