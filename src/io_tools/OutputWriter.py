from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import netCDF4
import numpy as np
import xarray as xr

import _version
from enums.forcings import Forcing
from kernel_wrappers.kernel_constants import EXIT_CODES

SOURCEFILE_GROUP_NAME = "sourcefile"
MODEL_DOMAIN_GROUP_NAME = "model_domain"
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
        skip_existing_file_checking: bool = False,
    ):
        """
        :param out_dir: directory to save outputfiles
        :param basename: base name of each outputfile (e.g. out_name = "3d_output" --> "3d_output_1993.nc")
        :param sourcefile: sourcefile, to be copied to outputfiles.
        :param forcing_data: xr.Datasets containing forcing datasets (e.g. currents, wind...)
        :param api_entry: the function where this is being called from (for traceability)
        :param api_arguments: dictionary containing info on the top-level API call
        """
        if out_dir.exists() and any(out_dir.iterdir()):
            print(
                f"DANGER: There are already files in '{out_dir}'! Contents may be overwritten!"
            )
            answer = "" if not skip_existing_file_checking else "y"
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
        self.forcing_meta = {
            forcing: xr.Dataset(ds.coords, attrs=ds.attrs)
            for forcing, ds in forcing_data.items()
        }
        self.api_entry = api_entry
        self.api_arguments = api_arguments

        self.model_domain = self._get_ocean_domain(forcing_data)

    @property
    @abstractmethod
    def _dataset_title(self) -> str:
        pass

    @property
    @abstractmethod
    def _group_names(self) -> List[str]:
        pass

    @abstractmethod
    def _get_ocean_domain(self, forcing_data: Dict[Forcing, xr.Dataset]) -> xr.Dataset:
        pass

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
            ds.title = self._dataset_title
            ds.description = (
                "This file's root group contains timeseries location data "
                "for particles run through ADVECTOR.  This file also contains "
                "several other self-describing groups: "
                f"{self._group_names}."
            )
            ds.institution = "The Ocean Cleanup"
            ds.source = f"ADVECTOR Version {_version.__version__}"
            ds.arguments = (
                f"The arguments of the call to {self.api_entry} which produced this "
                f"file are: {str(self.api_arguments)}"
            )

            ds.createDimension("p_id", len(chunk.p_id))
            ds.createDimension("time", None)  # unlimited dimension

            # Variables along only the static dimension, p_id
            p_id = ds.createVariable("p_id", chunk.p_id.dtype, ("p_id",))
            p_id[:] = chunk.p_id.values

            release_date = ds.createVariable("release_date", np.float64, ("p_id",))
            release_date.units = "seconds since 1970-01-01 00:00:00.0"
            release_date.calendar = "gregorian"
            release_date[:] = chunk.release_date.values.astype("datetime64[s]").astype(
                np.float64
            )

            exit_code = ds.createVariable("exit_code", np.byte, ("p_id",))
            exit_code.description = (
                "These codes are returned by the kernel when unexpected behavior occurs and the"
                "kernel must be terminated.  Their semantic meaning is provided in the "
                "'code_to_meaning' attribute of this variable."
            )
            exit_code.code_to_meaning = str(
                {code: meaning for code, meaning in EXIT_CODES.items() if code >= 0}
            )
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
        self.model_domain.to_netcdf(self.paths[-1], mode="a", group="model_domain")
        self.sourcefile.to_netcdf(self.paths[-1], mode="a", group=SOURCEFILE_GROUP_NAME)
        for forcing, meta in self.forcing_meta.items():
            meta.attrs["group_description"] = (
                f"This group contains the coordinates of the fully concatenated {forcing.value} "
                "dataset, after it has been loaded into ADVECTOR, and global attributes "
                "from the first file in the dataset."
            )
            meta.to_netcdf(self.paths[-1], mode="a", group=forcing.name + "_meta")

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
            time[start_t:] = chunk.time.values.astype("datetime64[s]").astype(
                np.float64
            )

            lon = ds.variables["lon"]
            lon[:, start_t:] = chunk.lon.values

            lat = ds.variables["lat"]
            lat[:, start_t:] = chunk.lat.values

            exit_code = ds.variables["exit_code"]
            # overwrite with most recent codes; by design, nonzero codes cannot change
            exit_code[:] = chunk.exit_code.values


class OutputWriter2D(OutputWriter):
    def _get_ocean_domain(self, forcing_data: Dict[Forcing, xr.Dataset]) -> xr.Dataset:
        land_mask_varname = "land_mask"
        return (
            (forcing_data[Forcing.current]["U"].isel(time=0).isnull())
            .drop_vars(("time", "depth"), errors="ignore")
            .rename(land_mask_varname)
            .assign_attrs(
                {
                    "description": "boolean mask which identifies 'land', defined as "
                    "the null cells in the sea surface current velocity field."
                }
            )
            .to_dataset()
            .assign_attrs(
                {
                    "domain_definition": "The internal model domain is defined as all grid cells which are "
                    f"not land; land is defined by the variable '{land_mask_varname}'."
                }
            )
        )

    @property
    def _dataset_title(self) -> str:
        return "Trajectories of Floating Marine Debris"

    @property
    def _group_names(self) -> List[str]:
        return [MODEL_DOMAIN_GROUP_NAME, SOURCEFILE_GROUP_NAME] + list(
            forcing.name + "_meta" for forcing in self.forcing_meta.keys()
        )


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
            radius = ds.createVariable("radius", np.float64, ("p_id",))
            radius.units = "meters"
            radius[:] = chunk.radius.values.astype(np.float64)

            density = ds.createVariable("density", np.float64, ("p_id",))
            density.units = "kg m^-3"
            density[:] = chunk.density.values.astype(np.float64)

            corey_shape_factor = ds.createVariable(
                "corey_shape_factor", np.float64, ("p_id",)
            )
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
            depth = ds.variables["depth"]
            depth[:, start_t:] = chunk.depth.values

    def _get_ocean_domain(self, forcing_data: Dict[Forcing, xr.Dataset]) -> xr.Dataset:
        return (
            forcing_data[Forcing.current]["bathymetry"]
            .assign_attrs(
                {
                    "description": "depth of ocean floor, defined as the bottom bound "
                    "of the first non-null cell of the ocean current "
                    "velocity field along the ascending depth dimension."
                }
            )
            .to_dataset()
            .assign_attrs(
                {
                    "domain_definition": "The internal model domain is defined as all points in "
                    "(lat, lon, depth) space where depth >= bathymetry, "
                    "excluding cells where bathymetry == 0."
                }
            )
        )

    @property
    def _dataset_title(self) -> str:
        return "Trajectories of Marine Debris"

    @property
    def _group_names(self) -> List[str]:
        return [
            MODEL_DOMAIN_GROUP_NAME,
            SOURCEFILE_GROUP_NAME,
            CONFIGFILE_GROUP_NAME,
        ] + list(forcing.name + "_meta" for forcing in self.forcing_meta.keys())
