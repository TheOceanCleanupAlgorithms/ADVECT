import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pandas as pd


def plot_advection(P, time, field, streamfunc=True, ax=None):
    # plot le advection
    if ax is None:
        _, ax = plt.subplots(1, 1)
    t_idx = -1
    dot, = ax.plot(P[:, 0, 0], P[:, 0, 1], '.', markersize=5)

    for i in range(len(time)):
        new_t_idx = np.argmin(np.abs(field.time - time[i]))
        if new_t_idx != t_idx and streamfunc:
            t_idx = new_t_idx
            ax.clear()
            dot, = ax.plot(P[:, i, 0], P[:, i, 1], '.', markersize=5)
            ax.streamplot(field.x, field.y, field.U[t_idx].T, field.V[t_idx].T)
        dot.set_xdata(P[:, i, 0])
        dot.set_ydata(P[:, i, 1])
        ax.set_title('t={:.2f}'.format(time[i]))
        ax.set_xlim([min(field.x), max(field.x)])
        ax.set_ylim([min(field.y), max(field.y)])
        plt.pause(.01)


def plot_ocean_advection(P, time: pd.DatetimeIndex):
    # plot le advection
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[14, 8])
    ax = plt.axes(projection=proj)
    ax.coastlines()

    dot, = ax.plot(P[:, 0, 0], P[:, 0, 1], '.')  # , transform=ccrs.Geodetic())

    for i in range(len(time)):
        dot.set_xdata(P[:, i, 0])
        dot.set_ydata(P[:, i, 1])
        ax.set_title(time[i])
        ax.set_ylim(-90, 90)
        plt.pause(.01)
