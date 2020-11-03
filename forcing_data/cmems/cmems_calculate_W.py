"""
script to calculate vertical ocean current velocity fields from the cmems_NP U/V
"""

import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import numpy as np

ds = xr.open_mfdataset("cmems_NP/*.nc").sel(depth=0, method="nearest").load()
U = ds["uo"]
V = ds["vo"]

diverge_v = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
diverge_u = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])

div = np.array([convolve(u, diverge_u, mode="constant") for u in U.values]) + np.array(
    [convolve(v, diverge_v, mode="constant") for v in V.values]
)

# downsample div
div = div.mean(axis=0)[1:, 1:].reshape([60, 10, 156, 10]).mean(-1).mean(1)
smooth = convolve(np.nan_to_num(div), 1 / 10 ** 2 * np.ones([10, 10]), mode="constant")
smooth[np.isnan(div)] = np.nan

plt.contourf(smooth, levels=np.linspace(-0.01, 0.01, 50), cmap=plt.get_cmap("seismic"))
plt.colorbar()
