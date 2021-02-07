import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from w_integrator import integrator


def load_ECCO():
    print("loading in the U/V dataset")
    U = xr.open_dataset('../examples/ECCO/ECCO_interp/U_2015-01-01.nc')
    V = xr.open_dataset('../examples/ECCO/ECCO_interp/V_2015-01-01.nc')
    W_true = xr.open_dataset('../examples/ECCO/ECCO_interp/W_2015-01-01.nc').W.squeeze()
    UV = xr.merge((U, V))
    UV = UV.sortby('depth', ascending=True).squeeze()
    # the coordinates of UV are assumed to specify the CENTERS of grid cells.  This can change between datasets...
    return UV, W_true


def calculate_W(UV):
    return integrator.generate_vertical_velocity(UV)


def compare_ECCO_gen_vs_given():
    UV, W_true = load_ECCO()
    W_calc = calculate_W(UV).sortby('depth', ascending=False)
    with ProgressBar():
        W_calc.load()

    # little interactive iteration through levels...
    print("Plotting...")
    for d in W_calc.depth:
        integrator.compare_Ws(("W_INTEGRATOR", W_calc), ("W_ECCO", W_true), depth=d, clip=5e-5)
        plt.pause(.01)
        input()
        plt.close()
