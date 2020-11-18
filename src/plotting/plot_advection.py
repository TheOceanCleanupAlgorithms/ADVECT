from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.animation as manimation
from tqdm import tqdm
import subprocess


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


def plot_ocean_trajectories(outputfile_path: str):
    P = xr.open_dataset(outputfile_path)
    # plot le advection
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[14, 8])
    ax = plt.axes(projection=proj)
    ax.coastlines()

    ax.plot(P.lon.transpose('time', 'p_id'), P.lat.transpose('time', 'p_id'), '.')


def plot_ocean_advection(outputfile_path: str, lon_range=(-180, 180), lat_range=(-90, 90)):
    P = xr.open_dataset(outputfile_path)
    # plot le advection
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[14, 8])
    ax = plt.axes(projection=proj)
    ax.coastlines()

    # invisible line forces map to at least cover the specified area, regardless of particle tracks
    ax.plot(np.linspace(*lon_range), np.linspace(*lat_range), alpha=0)

    # initialize the scatter plot with dummy data.
    trunc_winter = mcol.ListedColormap(cm.winter(np.linspace(0, .8, 100)))

    dot = ax.scatter(np.zeros(len(P.p_id)), np.zeros(len(P.p_id)), c=np.zeros(len(P.p_id)), cmap=trunc_winter,
                     s=5, norm=mcol.Normalize(vmin=P.depth.min(), vmax=P.depth.max()))
    cbar = plt.colorbar(mappable=dot, ax=ax, fraction=0.0235, pad=0.04)
    cbar.ax.set_ylabel('Depth (m)')

    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=30)
    outfile = Path(outputfile_path).with_suffix('.mp4')
    print("Creating Movie...")
    with writer.saving(fig, outfile=outfile, dpi=150):
        for i in tqdm(range(len(P.time))):
            dot.set_offsets(np.c_[np.array([P.isel(time=i).lon, P.isel(time=i).lat]).T])
            dot.set_array(P.isel(time=i).depth.values)
            ax.set_title(P.time.values[i])
            ax.set_ylim(-90, 90)
            writer.grab_frame()

    plt.close()
    print("Opening Movie...")
    subprocess.call(['open', outfile])  # this won't work except on mac.


@click.command()
@click.argument("outputfile_path", type=click.Path(exists=True, dir_okay=False, readable=True))
def plot_ocean_advection_CLI(outputfile_path: str):
    plot_ocean_advection(outputfile_path)


if __name__ == '__main__':
    plot_ocean_advection_CLI()
