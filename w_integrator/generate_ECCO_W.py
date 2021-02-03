import xarray as xr
import matplotlib.pyplot as plt

from w_integrator import integrator

print("loading in the U/V dataset")
U = xr.open_dataset('../examples/ECCO/ECCO_interp/U_2015-01-01.nc')
V = xr.open_dataset('../examples/ECCO/ECCO_interp/V_2015-01-01.nc')
W_true = xr.open_dataset('../examples/ECCO/ECCO_interp/W_2015-01-01.nc').W.squeeze()
UV = xr.merge((U, V))
UV = UV.sortby('depth', ascending=True).squeeze()
# the coordinates of UV are assumed to specify the CENTERS of grid cells.  This can change between datasets...


W_calc = integrator.generate_vertical_velocity(UV)
# little interactive iteration through levels...
print("Plotting...")
for i in range(0, 50):
    integrator.compare_Ws(("W_INTEGRATOR", W_calc), ("W_ECCO", W_true), level=i)
    plt.waitforbuttonpress()
    plt.close()
