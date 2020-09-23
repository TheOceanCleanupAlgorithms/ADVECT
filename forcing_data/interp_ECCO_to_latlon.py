import numpy as np
import xarray as xr
import ecco_v4_py as ecco
from tqdm import tqdm
import os
import glob

OUT_DIR = 'ECCO_interp/'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

new_grid_delta_lat = 1  # resolution of interpolated field (deg)
new_grid_delta_lon = 1
new_grid_min_lat = -90+new_grid_delta_lat/2  # domain of interpolated field (deg)
new_grid_max_lat = 90-new_grid_delta_lat/2
new_grid_min_lon = -180+new_grid_delta_lon/2
new_grid_max_lon = 180-new_grid_delta_lon/2

ECCO_grid = xr.open_dataset('./ECCO/ECCO-GRID.nc')

variables = {'EVEL': 'U',  # ECCO varname: [local varname, vertical grid name]
             'NVEL': 'V',
             'WVELMASS': 'W'}


for ECCO_varname, local_varname in variables.items():
    print(f'Interpolating all {ECCO_varname} files...')
    files = sorted(glob.glob(f'./ECCO/{ECCO_varname}*.nc'))
    for file in tqdm(files):
        ds = xr.open_dataset(file)

        if ECCO_varname == 'WVELMASS':  # interpolate vertical grid to Z from Zl; change vertical indexer name
            ds[ECCO_varname] = ds[ECCO_varname].assign_coords(k_l=ECCO_grid.Zl)
            ds[ECCO_varname] = ds[ECCO_varname].interp(k_l=ECCO_grid.Z.values)
            ds[ECCO_varname] = ds[ECCO_varname].assign_coords(k_l=ECCO_grid.k.values)
            ds[ECCO_varname] = ds[ECCO_varname].rename({'k_l': 'k'})
        interp_levels = []
        for lev in ds.k:
            new_grid_lon, new_grid_lat, var_interp =\
                ecco.resample_to_latlon(ds.XC, ds.YC,
                                        ds[ECCO_varname].sel(k=lev).where(ECCO_grid.maskC.sel(k=lev) > 0),  # mask land to nans
                                        new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,
                                        new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,
                                        fill_value=np.NaN, mapping_method='nearest_neighbor')
            interp_levels.append(var_interp)

        field_interpd_to_latlon = xr.DataArray(name=local_varname, data=np.array(interp_levels)[np.newaxis],
                                               coords=[ds.time.values, ECCO_grid.Z.values,
                                                       new_grid_lat[:, 0].astype('float32'),
                                                       new_grid_lon[0].astype('float32')],
                                               dims=['time', 'depth', 'lat', 'lon'])

        field_interpd_to_latlon.attrs = ds[ECCO_varname].attrs
        field_interpd_to_latlon.time.attrs = ds.time.attrs
        field_interpd_to_latlon.depth.attrs = ECCO_grid.Z.attrs
        field_interpd_to_latlon.lat.attrs = {'units': 'degrees_north', 'long_name': 'latitude'}
        field_interpd_to_latlon.lon.attrs = {'units': 'degrees_east', 'long_name': 'longitude'}

        field_interpd_to_latlon.to_netcdf(OUT_DIR + f'{local_varname}_{field_interpd_to_latlon.time.dt.strftime("%Y-%m-%d").values[0]}.nc')
