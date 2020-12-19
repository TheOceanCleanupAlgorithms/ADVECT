import xarray as xr


def get_eddy_diffusivity(configfile_path: str) -> xr.Dataset:
    return xr.open_dataset(configfile_path)
