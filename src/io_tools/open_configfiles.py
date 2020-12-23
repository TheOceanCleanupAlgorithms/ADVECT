import xarray as xr


def get_eddy_diffusivity(configfile_path: str) -> xr.Dataset:
    """load eddy diffusivity variables out of configfile, return as xr.Dataset"""
    eddy_diffusivity = xr.open_dataset(configfile_path)
    return eddy_diffusivity.sortby('z_hd', ascending=True)
