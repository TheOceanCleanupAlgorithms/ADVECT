from typing import Tuple

import xarray as xr

EDDY_DIFFUSIVITY_VARIABLES = {"z_hd", "horizontal_diffusivity", "z_vd", "vertical_diffusivity"}
OPTIONAL_VARIABLES_WITH_DEFAULTS = {
    "max_wave_height": 20,  # meters
    "wave_mixing_depth_factor": 10,  # unitless
}


def unpack_configfile(configfile_path: str) -> Tuple[xr.Dataset, float, float]:
    """
    :param configfile_path: path to configfile
    return: (eddy diffusivity, max wave height, wave mixing depth factor)
    """
    configfile = xr.open_dataset(configfile_path)
    for var in EDDY_DIFFUSIVITY_VARIABLES:
        assert var in configfile.variables, f"configfile {configfile_path} missing variable '{var}'"
    eddy_diffusivity = configfile[list(EDDY_DIFFUSIVITY_VARIABLES)]
    eddy_diffusivity = eddy_diffusivity.sortby(["z_hd", "z_vd"], ascending=True)

    if "max_wave_height" in configfile.variables:
        max_wave_height = float(configfile["max_wave_height"])
    else:
        max_wave_height = OPTIONAL_VARIABLES_WITH_DEFAULTS["max_wave_height"]

    if "wave_mixing_depth_factor" in configfile.variables:
        wave_mixing_depth_factor = float(configfile["wave_mixing_depth_factor"])
    else:
        wave_mixing_depth_factor = OPTIONAL_VARIABLES_WITH_DEFAULTS["wave_mixing_depth_factor"]

    return eddy_diffusivity, max_wave_height, wave_mixing_depth_factor
