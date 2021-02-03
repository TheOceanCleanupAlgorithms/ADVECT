import xarray as xr

from w_integrator import integrator

UV = xr.open_dataset('../examples/HYCOM_GLBu/hycom_GLBv_global_uv_2015-01-01T00.nc').squeeze()
verbose = True

