import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from w_integrator import integrator


def load_ECCO(u_path, v_path, w_path):
    print("loading in the U/V dataset")
    U = xr.open_dataset(u_path)
    V = xr.open_dataset(v_path)
    W_true = xr.open_dataset(w_path).W.squeeze()
    UV = xr.merge((U, V))
    UV = UV.sortby("depth", ascending=True).squeeze()
    # the coordinates of UV are assumed to specify the CENTERS of grid cells.  This can change between datasets...
    return UV, W_true


def compare_ECCO_gen_vs_given(u_path, v_path, w_path):
    UV, W_true = load_ECCO(u_path=u_path, v_path=v_path, w_path=w_path)
    W_calc = integrator.generate_vertical_velocity(UV).sortby("depth", ascending=False)
    with ProgressBar():
        W_calc.load()

    # little interactive iteration through levels...
    print("Plotting...")
    for d in W_calc.depth:
        integrator.compare_Ws(
            ("W_INTEGRATOR", W_calc), ("W_ECCO", W_true), depth=d, clip=None
        )
        plt.pause(0.01)
        input()
        plt.close()


if __name__ == "__main__":
    compare_ECCO_gen_vs_given(
        u_path="../examples/ECCO/ECCO_interp/U_2015-01-01.nc",
        v_path="../examples/ECCO/ECCO_interp/V_2015-01-01.nc",
        w_path="../examples/ECCO/ECCO_interp/W_2015-01-01.nc",
    )
