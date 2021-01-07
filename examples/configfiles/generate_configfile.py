import xarray as xr
import numpy as np
from pathlib import Path


def generate_configfile(
    horizontal_diffusivity: np.ndarray,
    z_hd: np.ndarray,
    vertical_diffusivity: np.ndarray,
    z_vd: np.ndarray,
    out_name: str,
):
    """
    script to generate a configuration file given requisite parameters
    :param horizontal_diffusivity: horizontal diffusivity at depth levels, m^2 s^-1
    :param z_hd: depth levels
    :param out_name: name at which to save config file
    """
    config = xr.Dataset(
        {
            "horizontal_diffusivity": (
                "z_hd",
                horizontal_diffusivity,
                {"units": "m^2 s^-1"},
            ),
            "vertical_diffusivity": (
                "z_vd",
                vertical_diffusivity,
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
            "z_vd": (
                "z_vd",
                z_vd,
                {
                    "long_name": "depth coordinate for vertical_diffusivity",
                    "units": "m",
                    "positive": "up",
                },
            ),
        },
        attrs={
            "title": f"Configuration File for ADVECTOR",
            "institution": "The Ocean Cleanup",
        },
    )

    out_path = Path(__file__).parent / out_name
    config.to_netcdf(out_path)


# a sample configuration file, profiles are NOT based on true ocean state
if __name__ == "__main__":
    generate_configfile(
        horizontal_diffusivity=np.linspace(1500, 1, 20),  # m^2 s^-1
        z_hd=-np.logspace(0, 4, 20),  # m
        vertical_diffusivity=np.linspace(-5e-3, 1e-2, 10)**2,
        z_vd=np.linspace(-1e4, 0, 10),  # m
        out_name="config.nc",
    )
