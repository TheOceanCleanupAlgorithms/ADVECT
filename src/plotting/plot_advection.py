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
from resources.fetch_etopo1 import fetch_etopo1


def plot_ocean_trajectories(outputfile_path: str):
    fig, ax = plt.subplots(figsize=[14, 8])

    # show current data grid
    bathy = fetch_etopo1().rename({'x': 'lon', 'y': 'lat', 'z': 'elevation'})
    xspacing = np.diff(bathy.lon).mean()
    yspacing = np.diff(bathy.lat).mean()
    ax.imshow(bathy.elevation <= 0, origin='lower', cmap='gray',
              extent=(bathy.lon.min()-xspacing/2, bathy.lon.max()+xspacing/2,
                      bathy.lat.min()-yspacing/2, bathy.lat.max()+yspacing/2))

    # plot trajectories
    P = xr.open_dataset(outputfile_path)
    ax.plot(P.lon.transpose('time', 'p_id'), P.lat.transpose('time', 'p_id'), '.')
    plt.show()


def animate_ocean_advection(outputfile_path: str, lon_range=(-180, 180), lat_range=(-90, 90), save: bool = False):
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
                     s=5, norm=mcol.Normalize(vmin=P.depth.min(), vmax=0))
    cbar = plt.colorbar(mappable=dot, ax=ax)
    cbar.ax.set_ylabel('Depth (m)')

    if save:
        animate_ocean_advection_to_disk(outputfile_path, P, fig, ax, dot)
    else:
        animate_ocean_advection_live(P, ax, dot)


def animate_ocean_advection_live(P, ax, dot):
    for i in range(len(P.time)):
        dot.set_offsets(np.c_[np.array([P.isel(time=i).lon, P.isel(time=i).lat]).T])
        dot.set_array(P.isel(time=i).depth.values)
        ax.set_title(P.time.values[i])
        ax.set_ylim(-90, 90)
        plt.pause(.005)
    plt.show()


def animate_ocean_advection_to_disk(outputfile_path, P, fig, ax, dot):
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
@click.option("-s", "--save_to_disk", is_flag=True)
@click.option("-t", "--trajectories", is_flag=True)
def plot_ocean_advection_CLI(outputfile_path: str, save_to_disk: bool, trajectories: bool):
    if trajectories:
        plot_ocean_trajectories(outputfile_path)
    else:
        animate_ocean_advection(outputfile_path, save=save_to_disk)


if __name__ == '__main__':
    plot_ocean_advection_CLI()
