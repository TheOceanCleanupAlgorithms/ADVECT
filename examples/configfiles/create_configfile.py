import xarray as xr
import numpy as np
from pathlib import Path


def create_configfile(horizontal_diffusivity: np.ndarray, z_hd: np.ndarray, out_path: Path):
    """
    :param horizontal_diffusivity: horizontal diffusivity at depth levels, m^2 s^-1
    :param z_hd: depth levels
    :param out_path: to save config file
    """
    config = xr.Dataset(
        {
            "horizontal_diffusivity": (
                "z_hd",
                horizontal_diffusivity,
                {"units": "m^2 s^-1"},
            ),
        },
        coords={
            "z_hd": (
                "z_hd",
                z_hd,
                {
                    "long_name": "depth coordinate for horizontal_diffusivity",
                    "units": "m",
                    "positive": "up",
                },
            ),
        },
        attrs={
            "title": "Configuration File for ADVECTOR",
            "institution": "The Ocean Cleanup",
        },
    )

    config.to_netcdf(out_path)


create_configfile(
    horizontal_diffusivity=np.linspace(1500, 1, 20),
    z_hd=-np.logspace(0, 4, 20),  # m
    out_path=Path(__file__).parent / "config.nc",
)
