import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from typing import Tuple


def generate_3D_sourcefile(
    num_particles: int,
    density_range: Tuple[float, float],
    radius_range: Tuple[float, float],
    corey_shape_factor_range: Tuple[float, float],
    depth_range: Tuple[float, float],
    release_date_range: Tuple[pd.Timestamp, pd.Timestamp],
    out_path: str,
    land_mask_path: str = Path(__file__).parent / "land_mask.nc",
):
    """
    generates a sourcefile with uniformly random properties within given bounds
    radius is distributed uniformly in log space
    :param num_particles: number of particles to generate
    :param density_range: density of particles, kg m^-3
    :param radius_range: radius of particles, m
    :param corey_shape_factor_range: unitless representation of 3d shape.  Domain: (.15, 1]
    :param depth_range: starting depth of particles, m (positive up)
    :param release_date_range: timestamp at which particle is released
    :param out_path: path at which to save file
    :param land_mask_path: path which contains a boolean land mask netcdf, dimensions {lon, lat}
    :return:
    """
    ds = create_2D_source_dataset(
        num_particles=num_particles,
        release_date_range=release_date_range,
        land_mask_path=land_mask_path,
    )
    rng = np.random.default_rng()
    ds = ds.assign(
        {
            "depth": ("p_id", rng.uniform(*depth_range, num_particles)),
            "radius": (
                "p_id",
                10 ** rng.uniform(*np.log10(radius_range), num_particles),
            ),
            "density": ("p_id", rng.uniform(*density_range, num_particles)),
            "corey_shape_factor": (
                "p_id",
                rng.uniform(*corey_shape_factor_range, num_particles),
            ),
        }
    )
    ds.attrs["title"] = f"3D Sourcefile for ADVECTOR"
    ds.to_netcdf(out_path)


def generate_2D_sourcefile(
    num_particles: int,
    release_date_range: Tuple[pd.Timestamp, pd.Timestamp],
    out_path: str,
    land_mask_path: str = Path(__file__).parent / "land_mask.nc",
):
    """
    generates a sourcefile with uniformly random properties within given bounds
    :param num_particles: number of particles to generate
    :param release_date_range: timestamp at which particle is released
    :param out_path: path at which to save file
    :param land_mask_path: path which contains a boolean land mask netcdf, dimensions {lon, lat}
    :return:
    """
    ds = create_2D_source_dataset(
        num_particles=num_particles,
        release_date_range=release_date_range,
        land_mask_path=land_mask_path,
    )
    ds.attrs["title"] = f"2D Sourcefile for ADVECTOR"
    ds.to_netcdf(out_path)


def create_2D_source_dataset(
    num_particles: int,
    release_date_range: Tuple[pd.Timestamp, pd.Timestamp],
    land_mask_path: str = Path(__file__).parent / "land_mask.nc",
):
    rng = np.random.default_rng()
    # create a land mask
    land = xr.open_dataarray(land_mask_path)

    # initialize particles
    [X, Y] = np.meshgrid(land.lon, land.lat)
    ocean_points = np.array([X[~land], Y[~land]]).T
    p0 = pd.DataFrame(
        data=rng.choice(ocean_points, size=num_particles, replace=True),
        columns=["lon", "lat"],
    )
    p0["release_date"] = pd.to_datetime(
        rng.uniform(
            release_date_range[0].timestamp(),
            release_date_range[1].timestamp(),
            num_particles,
        ),
        unit="s",
    )
    p0["p_id"] = np.arange(num_particles)
    ds = xr.Dataset(p0.set_index("p_id"))
    ds.attrs["institution"] = "The Ocean Cleanup"
    return ds


if __name__ == "__main__":
    generate_2D_sourcefile(
        num_particles=5000,
        release_date_range=(pd.Timestamp(2015, 1, 1), pd.Timestamp(2015, 12, 31)),
        out_path=str(Path(__file__).parent / "2D_uniform_source_2015.nc"),
    )
    generate_3D_sourcefile(
        num_particles=5000,
        density_range=(900, 1100),
        radius_range=(1e-6, 1e-1),
        corey_shape_factor_range=(0.15, 1),
        release_date_range=(pd.Timestamp(2015, 1, 1), pd.Timestamp(2015, 12, 31)),
        depth_range=(0, 0),
        out_path=str(Path(__file__).parent / "3D_uniform_source_2015.nc"),
    )
