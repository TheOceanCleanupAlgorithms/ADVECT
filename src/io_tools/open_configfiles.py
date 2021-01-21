import xarray as xr

EDDY_DIFFUSIVITY_VARIABLES = {"z_hd", "horizontal_diffusivity", "z_vd", "vertical_diffusivity"}
DENSITY_PROFILE_VARIABLES = {"z_sd", "seawater_density"}


def load_eddy_diffusivity(configfile_path: str) -> xr.Dataset:
    """load eddy diffusivity variables out of configfile, return as xr.Dataset"""
    configfile = xr.open_dataset(configfile_path)
    for var in EDDY_DIFFUSIVITY_VARIABLES:
        assert var in configfile.variables, f"configfile {configfile_path} missing variable '{var}'"
    eddy_diffusivity = configfile[list(EDDY_DIFFUSIVITY_VARIABLES)]
    return eddy_diffusivity.sortby(["z_hd", "z_vd"], ascending=True)


def load_density_profile(configfile_path: str) -> xr.Dataset:
    configfile = xr.open_dataset(configfile_path)
    for var in DENSITY_PROFILE_VARIABLES:
        assert var in configfile.variables, f"configfile {configfile_path} missing variable '{var}'"
    density_profile = configfile[list(DENSITY_PROFILE_VARIABLES)]
    return density_profile.sortby(["z_sd"], ascending=True)
