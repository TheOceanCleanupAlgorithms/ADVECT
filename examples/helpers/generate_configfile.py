from pathlib import Path

import numpy as np
import xarray as xr


def generate_configfile(
    out_path: Path,
    horizontal_diffusivity: np.ndarray,
    z_hd: np.ndarray,
    vertical_diffusivity: np.ndarray,
    z_vd: np.ndarray,
    max_wave_height: float = None,
    wave_mixing_depth_factor: float = None,
):
    """
    script to generate a configuration file given requisite parameters
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

    if max_wave_height:
        config["max_wave_height"] = max_wave_height
    if wave_mixing_depth_factor:
        config["wave_mixing_depth_factor"] = wave_mixing_depth_factor

    config.to_netcdf(out_path)


def generate_sample_configfile(out_path: Path):
    """a sample configuration file, diffusivity profiles are NOT based on true ocean state"""
    generate_configfile(
        horizontal_diffusivity=np.linspace(1500, 1, 20),  # m^2 s^-1
        z_hd=-np.logspace(0, 4, 20),  # m
        vertical_diffusivity=np.linspace(-5e-3, 1e-2, 10) ** 2,
        z_vd=np.linspace(-1e4, 0, 10),  # m
        out_path=out_path,
    )


if __name__ == "__main__":
    generate_sample_configfile(out_path=Path(__file__).parent / "sample_config.nc")
