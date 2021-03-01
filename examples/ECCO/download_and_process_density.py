import xarray as xr
from examples.ECCO.interp_ECCO_to_latlon import interpolate_variable


def download_density():
    print(
        """
        To download seawater_density, you can get monthly files here:
        https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/nctiles_monthly/RHOAnoma/2015.
        Download them to ./ECCO_native.  This is quite easy so an automated script is not provided.
        """
    )


def process_density():
    """Assumes data is downloaded into ./ECCO_nativ"""
    # interpolate to a lat/lon grid
    new_grid_delta_lat = 1  # resolution of interpolated field (deg)
    new_grid_delta_lon = 1
    new_grid_min_lat = (
        -90 + new_grid_delta_lat / 2
    )  # domain of interpolated field (deg)
    new_grid_max_lat = 90 - new_grid_delta_lat / 2
    new_grid_min_lon = -180 + new_grid_delta_lon / 2
    new_grid_max_lon = 180 - new_grid_delta_lon / 2
    interpolate_variable(
        "RHOAnoma",
        "RHOAnoma",
        new_grid_min_lat,
        new_grid_max_lat,
        new_grid_delta_lat,
        new_grid_min_lon,
        new_grid_max_lon,
        new_grid_delta_lon,
    )

    rhoConst = 1029  # source: https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/nctiles_monthly/README
    rho_anom = xr.open_mfdataset("./ECCO_interp/RHOAnoma_2015*.nc")["RHOAnoma"]
    rho_abs = rhoConst + rho_anom
    rho_abs.name = "rho"
    rho_abs.attrs["long_name"] = "Seawater Density"
    rho_abs.to_netcdf("./ECCO_interp/RHO_2015.nc")


if __name__ == "__main__":
    process_density()
