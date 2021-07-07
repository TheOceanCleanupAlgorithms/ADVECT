import subprocess
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.animation as manimation
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm


def plot_ocean_trajectories(outputfile_path: str):
    land_mask = xr.open_dataset(outputfile_path, group="model_domain").land_mask
    fig, ax = plt.subplots(figsize=[14, 8])

    xspacing = np.diff(land_mask.lon).mean()
    yspacing = np.diff(land_mask.lat).mean()
    lon_edges = np.append(
        (land_mask.lon - xspacing / 2), land_mask.lon[-1] + xspacing / 2
    )
    lat_edges = np.append(
        (land_mask.lat - yspacing / 2), land_mask.lat[-1] + yspacing / 2
    )
    plt.pcolormesh(lon_edges, lat_edges, ~land_mask, cmap="gray")

    # plot trajectories
    P = xr.open_dataset(outputfile_path)
    ax.plot(P.lon.transpose("time", "p_id"), P.lat.transpose("time", "p_id"), ".")
    plt.show()


def animate_ocean_advection(
    outputfile_path: str,
    lon_range=(-180, 180),
    lat_range=(-90, 90),
    save: bool = False,
    colorbar_depth=None,
):
    P = xr.open_dataset(outputfile_path)
    # plot le advection
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[14, 8])
    ax = plt.axes(projection=proj)
    ax.coastlines()

    # invisible line forces map to at least cover the specified area, regardless of particle tracks
    ax.plot(np.linspace(*lon_range), np.linspace(*lat_range), alpha=0)

    # initialize the scatter plot with dummy data.
    trunc_winter = mcol.ListedColormap(cm.winter(np.linspace(0, 0.8, 100)))
    if "depth" in P.variables:
        vmin = colorbar_depth if colorbar_depth is not None else P.depth.min()
        dot = ax.scatter(
            np.zeros(len(P.p_id)),
            np.zeros(len(P.p_id)),
            c=np.zeros(len(P.p_id)),
            cmap=trunc_winter,
            s=5,
            norm=mcol.Normalize(vmin=vmin, vmax=0),
        )
        cbar = plt.colorbar(mappable=dot, ax=ax)
        cbar.ax.set_ylabel("Depth (m)")
    else:
        dot = ax.scatter(
            np.zeros(len(P.p_id)), np.zeros(len(P.p_id)), c="tab:blue", s=5
        )

    def base_update(i, P, ax, dot, timestr):
        dot.set_offsets(np.c_[np.array([P.isel(time=i).lon, P.isel(time=i).lat]).T])
        ax.set_title(timestr[i])
        ax.set_ylim(-90, 90)

    if "depth" in P.variables:

        def update_func(i, P, ax, dot, timestr):
            base_update(i, P, ax, dot, timestr)
            dot.set_array(P.isel(time=i).depth.values)

    else:
        update_func = base_update

    timestr = P.time.dt.strftime("%Y-%m-%dT%H:%M:%S").values
    if save:
        animate_ocean_advection_to_disk(
            outputfile_path, P, fig, ax, dot, timestr, update_func
        )
    else:
        for i in range(len(P.time)):
            update_func(i, P, ax, dot, timestr)
            plt.pause(0.005)
        plt.show()


def animate_ocean_advection_to_disk(
    outputfile_path, P, fig, ax, dot, timestr, update_func
):
    FFMpegWriter = manimation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=30)
    outfile = Path(outputfile_path).with_suffix(".mp4")
    print("Creating Movie...")
    with writer.saving(fig, outfile=outfile, dpi=150):
        for i in tqdm(range(len(P.time))):
            update_func(i, P, ax, dot, timestr)
            writer.grab_frame()

    plt.close()
    print("Opening Movie...")
    subprocess.call(["open", outfile])  # this won't work except on mac.
