import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# load some data
U = xr.open_mfdataset(sorted(glob.glob('ECCO/ECCO_interp/U_2015*.nc')))
V = xr.open_mfdataset(sorted(glob.glob('ECCO/ECCO_interp/V_2015*.nc')))
W = xr.open_mfdataset(sorted(glob.glob('ECCO/ECCO_interp/W_2015*.nc')))

currents = xr.merge((U, V, W))

[X, Y] = np.meshgrid(currents.lon, currents.lat)

plt.figure(figsize=(15, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
field = currents.W.isel(depth=0).mean(dim='time')
absmax = np.max(np.abs(field))
cf = ax.contourf(X, Y, field, cmap='seismic', levels=np.linspace(-absmax/5, absmax/5, 20))
plt.colorbar(mappable=cf, ax=ax, shrink=.75)
ax.coastlines()
plt.title('Surface Upwelling 2015 time-mean (m/s)')
plt.tight_layout()

fig = plt.figure(figsize=(15, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
U = currents.U.isel(depth=0, time=0).values
V = currents.V.isel(depth=0, time=0).values
cf = ax.contourf(X, Y, (U**2 + V**2)**.5, cmap='viridis', levels=20)
plt.colorbar(mappable=cf, ax=ax, shrink=.75)
skip = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(X[skip], Y[skip], U[skip], V[skip], scale=40, width=.001)
ax.coastlines()
plt.title(f'Surface currents {currents.time[0].dt.strftime("%Y-%m-%d").values} (m/s)')
plt.tight_layout()
