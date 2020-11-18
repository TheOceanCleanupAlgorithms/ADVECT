from pathlib import Path

import click
import xarray as xr
import pandas as pd
import numpy as np
import os


@click.command()
@click.option('-n', 'num_particles', required=True, type=click.INT)
@click.option('-o', 'out_path', required=False, type=click.Path(exists=False, dir_okay=False),
              default=os.path.join(os.path.dirname(__file__), '2015_uniform_two_releases.nc'))
def generate_sourcefile(
    num_particles: int,
    out_path: str,
):
    """
    :param out_path: path to save the sourcefile at
    :param num_particles: number of particles to generate
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
    #                     np.concatenate((np.full(num_particles//2, np.datetime64('2015-01-01T12')),
    #                                    np.full(num_particles//2 + num_particles % 2, np.datetime64('2015-06-01'))))
    p0['depth'] = 0

    p0['p_id'] = np.arange(num_particles)
    ds = xr.Dataset(p0.set_index('p_id'))

    ds.to_netcdf(out_path)


if __name__ == '__main__':
    generate_sourcefile()
