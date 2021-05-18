from abc import ABC
from pathlib import Path
from typing import Dict

import netCDF4
import numpy as np
import xarray as xr

from _version import __version__
from enums.forcings import Forcing
from kernel_wrappers.kernel_constants import EXIT_CODES

SOURCEFILE_GROUP_NAME = "sourcefile"
CONFIGFILE_GROUP_NAME = "configfile"


class OutputWriter(ABC):
    def __init__(
        self,
        out_dir: Path,
        basename: str,
        sourcefile: xr.Dataset,
        forcing_data: Dict[Forcing, xr.Dataset],
        api_entry: str,
        api_arguments: dict,
    ):
        """
        :param out_dir: directory to save outputfiles
        :param basename: base name of each outputfile (e.g. out_name = "3d_output" --> "3d_output_1993.nc")
        :param sourcefile: sourcefile, to be copied to outputfiles.
        :param forcing_data: xr.Datasets containing forcing datasets (e.g. currents, wind...)
        :param api_arguments: dictionary containing info on the top-level API call
        """
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"DANGER: There are already files in '{out_dir}'! Contents may be overwritten!")
            answer = ""
            while answer not in {"y", "n"}:
                answer = input("Continue anyway? [y/n]: ")
            if answer == "n":
                raise SystemExit
        out_dir.mkdir(exist_ok=True, parents=True)

        self.folder_path = out_dir
        self.basename = basename
        self.current_year = None
        self.paths = []

        self.sourcefile = sourcefile
        self.forcing_meta = {forcing: xr.Dataset(ds.coords, attrs=ds.attrs) for forcing, ds in forcing_data.items()}
        self.api_entry = api_entry
        self.api_arguments = api_arguments

    def _set_current_year(self, year: int):
        self.current_year = year
        self.paths.append(self.folder_path / f"{self.basename}_{year}.nc")

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
            # --- INITIALIZE PARTICLE TRAJECTORIES IN ROOT GROUP --- #
            ds.institution = "The Ocean Cleanup"
            ds.source = f"ADVECTOR Version {__version__}"
            ds.arguments = f"The arguments of the call to {self.api_entry} which produced this " \
                           f"file are: {str(self.api_arguments)}"

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

        # --- SAVE MODEL CONFIGURATION METADATA INTO GROUPS --- #
        self.sourcefile.to_netcdf(self.paths[-1], mode="a", group=SOURCEFILE_GROUP_NAME)
        for forcing, meta in self.forcing_meta.items():
            meta.attrs["group_description"] = (
                f"This group contains the coordinates of the fully concatenated {forcing.value} "
                "dataset, after it has been loaded into ADVECTOR, and global attributes "
                "from the first file in the dataset."
            )
            meta.to_netcdf(self.paths[-1], mode="a", group=forcing.name+"_meta")

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
            ds.title = "Trajectories of Floating Marine Debris"
            ds.description = "This file's root group contains timeseries location data for a batch of particles run " \
                             "through ADVECTOR.  This file also contains several other groups: " \
                             f"{SOURCEFILE_GROUP_NAME}, which is a copy of the sourcefile passed to ADVECTOR, " \
                             f"and a group for each forcing dataset: {list(forcing.name+'_meta' for forcing in self.forcing_meta.keys())}, " \
                             f"which each contain the dataset's coordinates " \
                             f"and the global attributes from the first file in the dataset."


class OutputWriter3D(OutputWriter):
    def __init__(
        self,
        out_dir: Path,
        basename: str,
        configfile: xr.Dataset,
        sourcefile: xr.Dataset,
        forcing_data: Dict[Forcing, xr.Dataset],
        api_entry: str,
        api_arguments: dict,
    ):
        """
        :param configfile: configfile, to be copied to outputfiles
        see OutputWriter for other arg descriptions
        """
        super().__init__(
            out_dir=out_dir,
            basename=basename,
            sourcefile=sourcefile,
            forcing_data=forcing_data,
            api_entry=api_entry,
            api_arguments=api_arguments,
        )
        self.configfile = configfile

    def _write_first_chunk(self, chunk: xr.Dataset):
        super()._write_first_chunk(chunk=chunk)

        # --- SAVE MODEL CONFIGURATION METADATA INTO GROUPS --- #
        self.configfile.to_netcdf(self.paths[-1], mode="a", group=CONFIGFILE_GROUP_NAME)

        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            # --- INITIALIZE PARTICLE TRAJECTORIES IN ROOT GROUP --- #
            ds.title = "Trajectories of Marine Debris"
            ds.description = "This file's root group contains timeseries location data for a batch of particles run " \
                             "through ADVECTOR.  This file also contains several other groups: " \
                             f"{CONFIGFILE_GROUP_NAME}, which is a copy of the configfile passed to ADVECTOR, " \
                             f"{SOURCEFILE_GROUP_NAME}, which is a copy of the sourcefile passed to ADVECTOR, " \
                             f"and a group for each forcing dataset: {list(forcing.name+'_meta' for forcing in self.forcing_meta.keys())}, " \
                             f"which each contain the dataset's coordinates " \
                             f"and the global attributes from the first file in the dataset."

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
