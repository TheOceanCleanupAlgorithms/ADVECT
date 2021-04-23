import os
from pathlib import Path
from typing import Optional

import xarray as xr
import netCDF4
import numpy as np

from kernel_wrappers.kernel_constants import EXIT_CODES
from _version import __version__

SOURCEFILE_GROUP_NAME = "sourcefile"
CONFIGFILE_GROUP_NAME = "configfile"
CURRENTS_META_GROUP_NAME = "currents_meta"
WIND_META_GROUP_NAME = "wind_meta"


class OutputWriter:
    def __init__(self, out_dir: Path, sourcefile_path: str,
                 currents: xr.Dataset, wind: Optional[xr.Dataset], arguments_to_run_advector: dict):
        """
        :param out_dir: directory to save outputfiles
        :param sourcefile_path: path to sourcefile
        :param currents: dataset containing the ocean currents
        :param wind: dataset containing the winds
        :param arguments_to_run_advector: dictionary containing info on the top-level API call, "run_advector.py::run_advector"
        """
        if not out_dir.is_dir():
            out_dir.mkdir()

        self.folder_path = out_dir
        self.current_year = None
        self.paths = []

        self.sourcefile_path = sourcefile_path
        self.currents_meta = xr.Dataset(currents.coords, attrs=currents.attrs)  # extract just coords and attributes
        self.wind_meta = xr.Dataset(wind.coords, attrs=wind.attrs) if wind is not None else None
        self.arguments_to_run_advector = arguments_to_run_advector

    def _set_current_year(self, year: int):
        self.current_year = year
        self.paths.append(self.folder_path / f"advector_output_{year}.nc")

    def write_output_chunk(self, chunk: xr.Dataset):
        beginning_year = chunk.time.dt.year.values[0]
        end_year = chunk.time.dt.year.values[-1]

        for year in range(beginning_year, end_year + 1):
            chunk_year = chunk.isel({"time": chunk.time.dt.year == year})
            if year != self.current_year:
                self._set_current_year(year)
                self._write_first_chunk(chunk_year)
                self._copy_unexpected_variables(chunk_year)
            else:
                self._append_chunk(chunk_year)

    def _write_first_chunk(self, chunk: xr.Dataset):
        with netCDF4.Dataset(self.paths[-1], mode="w") as ds:
            # --- SAVE MODEL CONFIGURATION METADATA INTO GROUPS --- #
            sourcefile_group = ds.createGroup(SOURCEFILE_GROUP_NAME)
            with netCDF4.Dataset(self.sourcefile_path, mode="r") as sourcefile:
                copy_dataset(sourcefile, sourcefile_group)

            currents_meta_group = ds.createGroup(CURRENTS_META_GROUP_NAME)
            currents_meta_group.setncattr(
                "currents_meta_group_description",
                "This group contains the coordinates of the fully concatenated currents "
                "dataset, after it has been loaded into ADVECTOR, and global attributes "
                "from the first zonal current file in the dataset."
            )

            tmp_currents_path = self.folder_path / "currents_meta_tmp.nc"
            self.currents_meta.to_netcdf(tmp_currents_path)  # save xr.Dataset to temp file
            with netCDF4.Dataset(tmp_currents_path, mode="r") as currents_meta:  # so we can open it with netCDF4
                copy_dataset(currents_meta, currents_meta_group)
            os.remove(tmp_currents_path)

            wind_meta_group = ds.createGroup(WIND_META_GROUP_NAME)
            wind_meta_group.setncattr(
                "wind_meta_group_description",
                "This group contains the coordinates of the fully concatenated 10-meter wind "
                "dataset, after it has been loaded into ADVECTOR, and global attributes "
                "from the first zonal wind file in the dataset."
            )
            if self.wind_meta is not None:
                tmp_wind_path = self.folder_path / "wind_meta_tmp.nc"
                self.wind_meta.to_netcdf(tmp_wind_path)  # save xr.Dataset to temp file
                with netCDF4.Dataset(tmp_wind_path, mode="r") as wind_meta:  # so we can open it with netCDF4
                    copy_dataset(wind_meta, wind_meta_group)
                os.remove(tmp_wind_path)

            # --- INITIALIZE PARTICLE TRAJECTORIES IN ROOT GROUP --- #
            ds.institution = "The Ocean Cleanup"
            ds.source = f"ADVECTOR Version {__version__}"
            ds.arguments = f"The arguments of the call to src/run_advector.py::run_advector which produced this " \
                           f"file are: {str(self.arguments_to_run_advector)}"

            ds.createDimension("p_id", len(chunk.p_id))
            ds.createDimension("time", None)  # unlimited dimension

            # Variables along only the static dimension, p_id
            p_id = ds.createVariable("p_id", chunk.p_id.dtype, ("p_id",))
            p_id[:] = chunk.p_id.values

            release_date = ds.createVariable("release_date", np.float64, ("p_id",))
            release_date.units = "seconds since 1970-01-01 00:00:00.0"
            release_date.calendar = "gregorian"
            release_date[:] = chunk.release_date.values.astype("datetime64[s]").astype(np.float64)

            exit_code = ds.createVariable("exit_code", np.byte, ("p_id",))
            exit_code.description = "These codes are returned by the kernel when unexpected behavior occurs and the" \
                                    "kernel must be terminated.  Their semantic meaning is provided in the " \
                                    "'code_to_meaning' attribute of this variable."
            exit_code.code_to_meaning = str({code: meaning for code, meaning in EXIT_CODES.items() if code >= 0})
            exit_code[:] = chunk.exit_code.values.astype(np.byte)

            # Variables that expand between chunks
            time = ds.createVariable("time", np.float64, ("time",))
            time.units = "seconds since 1970-01-01 00:00:00.0"
            time.calendar = "gregorian"
            time[:] = chunk.time.values.astype("datetime64[s]").astype(np.float64)

            lon = ds.createVariable("lon", chunk.lon.dtype, ("p_id", "time"))
            lon.units = "Degrees East"
            lon[:] = chunk.lon.values

            lat = ds.createVariable("lat", chunk.lat.dtype, ("p_id", "time"))
            lat.units = "Degrees North"
            lat[:] = chunk.lat.values

    def _copy_unexpected_variables(self, chunk: xr.Dataset):
        """copy any variables along only p_id should be copied over as well"""
        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            dims_except_p_id = [dim for dim in chunk.dims if dim != "p_id"]
            for varname in chunk.drop_dims(dims_except_p_id).variables:
                if varname not in ds.variables:
                    ds.createVariable(varname, chunk[varname].dtype, ("p_id",))
                    ds[varname].setncatts(chunk[varname].attrs)
                    ds[varname][:] = chunk[varname].values

    def _append_chunk(self, chunk: xr.Dataset):
        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            time = ds.variables["time"]
            start_t = len(time)
            time[start_t:] = chunk.time.values.astype("datetime64[s]").astype(np.float64)

            lon = ds.variables["lon"]
            lon[:, start_t:] = chunk.lon.values

            lat = ds.variables["lat"]
            lat[:, start_t:] = chunk.lat.values

            exit_code = ds.variables["exit_code"]
            # overwrite with most recent codes; by design, nonzero codes cannot change
            exit_code[:] = chunk.exit_code.values


class OutputWriter2D(OutputWriter):
    def _write_first_chunk(self, chunk: xr.Dataset):
        super()._write_first_chunk(chunk)
        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            ds.title = "Trajectories of Marine Debris"
            ds.description = "This file's root group contains timeseries location data for a batch of particles run " \
                             "through ADVECTOR.  This file also contains several other groups: " \
                             f"{SOURCEFILE_GROUP_NAME}, which is a copy of the sourcefile passed to ADVECTOR, " \
                             f"{CURRENTS_META_GROUP_NAME}, which contains the coordinates of the current dataset passed " \
                             f"to ADVECTOR, as well as the global attributes from the first zonal current file, and " \
                             f"{WIND_META_GROUP_NAME}, which contains the coordinates of the wind dataset passed to " \
                             f"ADVECTOR, as well as the global attributes from the first zonal wind file."


class OutputWriter3D(OutputWriter):
    def __init__(self, out_dir: Path, configfile_path: str, sourcefile_path: str,
                 currents: xr.Dataset, wind: Optional[xr.Dataset], arguments_to_run_advector: dict):
        """
        :param out_dir: directory to save outputfiles
        :param configfile_path: path to configfile
        :param sourcefile_path: path to sourcefile
        :param currents: dataset containing the ocean currents
        :param wind: dataset containing the winds
        :param arguments_to_run_advector: dictionary containing info on the top-level API call, "run_advector.py::run_advector"
        """
        super().__init__(
            out_dir=out_dir,
            sourcefile_path=sourcefile_path,
            currents=currents, wind=wind,
            arguments_to_run_advector=arguments_to_run_advector,
        )
        self.configfile_path = configfile_path

    def _write_first_chunk(self, chunk: xr.Dataset):
        super()._write_first_chunk(chunk=chunk)
        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            # --- SAVE MODEL CONFIGURATION METADATA INTO GROUPS --- #
            config_group = ds.createGroup("configfile")
            with netCDF4.Dataset(self.configfile_path, mode="r") as configfile:
                copy_dataset(configfile, config_group)

            # --- INITIALIZE PARTICLE TRAJECTORIES IN ROOT GROUP --- #
            ds.title = "Trajectories of Marine Debris"
            ds.description = "This file's root group contains timeseries location data for a batch of particles run " \
                             "through ADVECTOR.  This file also contains several other groups: " \
                             f"{CONFIGFILE_GROUP_NAME}, which is a copy of the configfile passed to ADVECTOR, " \
                             f"{SOURCEFILE_GROUP_NAME}, which is a copy of the sourcefile passed to ADVECTOR, " \
                             f"{CURRENTS_META_GROUP_NAME}, which contains the coordinates of the current dataset passed " \
                             f"to ADVECTOR, as well as the global attributes from the first zonal current file, and " \
                             f"{WIND_META_GROUP_NAME}, which contains the coordinates of the wind dataset passed to " \
                             f"ADVECTOR, as well as the global attributes from the first zonal wind file."

            radius = ds.createVariable("radius", np.float64, ("p_id",))
            radius.units = "meters"
            radius[:] = chunk.radius.values.astype(np.float64)

            density = ds.createVariable("density", np.float64, ("p_id",))
            density.units = "kg m^-3"
            density[:] = chunk.density.values.astype(np.float64)

            corey_shape_factor = ds.createVariable("corey_shape_factor", np.float64, ("p_id",))
            corey_shape_factor.units = "unitless"
            corey_shape_factor[:] = chunk.corey_shape_factor.values.astype(np.float64)

            depth = ds.createVariable("depth", chunk.depth.dtype, ("p_id", "time"))
            depth.units = "meters"
            depth.positive = "up"
            depth[:] = chunk.depth.values

    def _append_chunk(self, chunk: xr.Dataset):
        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            start_t = len(ds.variables["time"])
        super()._append_chunk(chunk=chunk)
        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            depth = ds.variables['depth']
            depth[:, start_t:] = chunk.depth.values


def copy_dataset(source: netCDF4.Dataset, destination: netCDF4.Dataset):
    """
    copy the contents (attributes, dimensions, variables) of 'source' into 'destination.'
    adapted from https://stackoverflow.com/a/49592545
    """
    # copy global attributes
    destination.setncatts(source.__dict__)
    # copy dimensions
    for name, dimension in source.dimensions.items():
        destination.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
    # copy variables
    for name, variable in source.variables.items():
        destination.createVariable(name, variable.datatype, variable.dimensions)
        # copy variable attributes
        destination[name].setncatts(source[name].__dict__)
        # copy variable contents
        destination[name][:] = source[name][:]
