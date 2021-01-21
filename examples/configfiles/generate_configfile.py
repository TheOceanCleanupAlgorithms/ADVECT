import xarray as xr
import numpy as np
import scipy.interpolate
from pathlib import Path


def generate_configfile(
    horizontal_diffusivity: np.ndarray,
    z_hd: np.ndarray,
    vertical_diffusivity: np.ndarray,
    z_vd: np.ndarray,
    seawater_density: np.ndarray,
    z_sd: np.ndarray,
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
            "horizontal_diffusivity": ("z_hd", horizontal_diffusivity,
                                       {"units": "m^2 s^-1"}),
            "vertical_diffusivity": ("z_vd", vertical_diffusivity,
                                     {"units": "m^2 s^-1"}),
            "seawater_density": ("z_sd", seawater_density,
                                 {"units": "kg m^-3"}),
        },
        coords={
            "z_hd": ("z_hd", z_hd,
                     {"long_name": "depth coordinate for horizontal_diffusivity",
                      "units": "m", "positive": "up"}),
            "z_vd": ("z_vd", z_vd,
                     {"long_name": "depth coordinate for vertical_diffusivity",
                      "units": "m", "positive": "up"}),
            "z_sd": ("z_sd", z_sd,
                     {"long_name": "depth coordinate for seawater_density",
                      "units": "m", "positive": "up"}),
        },
        attrs={
            "title": f"Configuration File for ADVECTOR",
            "institution": "The Ocean Cleanup",
        },
    )

    out_path = Path(__file__).parent / out_name
    config.to_netcdf(out_path)


# a sample configuration file, diffusivity profiles are NOT based on true ocean state
if __name__ == "__main__":
    # global lat/lon/time average of ocean density, calculated from HYCOM 2015 monthly Temp/Salinity data
    density_profile = np.array(  # kg m^-3
        [1024.68447813, 1024.70283848, 1024.71876261, 1024.73270559,
         1024.74796112, 1024.76149485, 1024.78942640, 1024.81759160,
         1024.87133391, 1024.92742960, 1024.98616821, 1025.05184627,
         1025.12685215, 1025.20513533, 1025.28762645, 1025.46287398,
         1025.63547038, 1025.79906481, 1025.97416052, 1026.14300119,
         1026.51278096, 1026.81962176, 1027.32070224, 1027.72365369,
         1028.07358366, 1028.39577868, 1028.70253710, 1029.28823021,
         1029.84525524, 1030.38609663, 1030.92039173, 1031.45174062,
         1031.97704453, 1033.25864804, 1034.50376827, 1036.91242367,
         1039.25936670, 1041.54760658, 1046.05280992, 1050.47720347]
    )
    density_profile_depth = np.array(  # m, positive down
        [0.00e+00, 2.00e+00, 4.00e+00, 6.00e+00, 8.00e+00, 1.00e+01,
         1.20e+01, 1.50e+01, 2.00e+01, 2.50e+01, 3.00e+01, 3.50e+01,
         4.00e+01, 4.50e+01, 5.00e+01, 6.00e+01, 7.00e+01, 8.00e+01,
         9.00e+01, 1.00e+02, 1.25e+02, 1.50e+02, 2.00e+02, 2.50e+02,
         3.00e+02, 3.50e+02, 4.00e+02, 5.00e+02, 6.00e+02, 7.00e+02,
         8.00e+02, 9.00e+02, 1.00e+03, 1.25e+03, 1.50e+03, 2.00e+03,
         2.50e+03, 3.00e+03, 4.00e+03, 5.00e+03]
    )
    # extrapolate profile down to maximum ocean depth.  Profile is pretty linear at deep depths so this seems fine
    density_interp = scipy.interpolate.interp1d(density_profile_depth, density_profile, fill_value='extrapolate')
    density_profile_depth = np.append(density_profile_depth, 11e3)  # 11000m, challenger deep (ish)
    density_profile = np.append(density_profile, density_interp(density_profile_depth[-1]))

    generate_configfile(
        horizontal_diffusivity=np.linspace(1500, 1, 20),  # m^2 s^-1
        z_hd=-np.logspace(0, 4, 20),  # m
        vertical_diffusivity=np.linspace(-5e-3, 1e-2, 10) ** 2,
        z_vd=np.linspace(-1e4, 0, 10),  # m
        seawater_density=density_profile,
        z_sd=-density_profile_depth,
        out_name="config.nc",
    )
