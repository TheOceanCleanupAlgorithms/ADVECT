import click
import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path


@click.command()
@click.option('-n', 'num_particles', required=False, default=5000)
@click.option('-o', 'out_name', required=False, type=click.Path(exists=False, dir_okay=False),
              default='2D_uniform_source.nc')
def generate_2D_sourcefile(
    num_particles: int,
    out_name: str,
):
    """
    :param num_particles: number of particles to generate
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

    p0['p_id'] = np.arange(num_particles)
    ds = xr.Dataset(p0.set_index('p_id'))

    ds.attrs["title"] = f"3D Sourcefile for ADVECTOR"
    ds.attrs["institution"] = "The Ocean Cleanup"

    out_path = Path(__file__).parent / out_name
    ds.to_netcdf(out_path)


if __name__ == '__main__':
    generate_2D_sourcefile()
