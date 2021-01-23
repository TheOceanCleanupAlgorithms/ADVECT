from pathlib import Path
import xarray as xr
import netCDF4
import numpy as np

from kernel_wrappers.kernel_constants import EXIT_CODES
from _version import __version__


class OutputWriter:
    def __init__(self, out_dir: Path, configfile_path: Path):
        if not out_dir.is_dir():
            out_dir.mkdir()

        self.folder_path = out_dir
        self.current_year = None
        self.paths = []

        self.configfile_path = configfile_path

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
            else:
                self._append_chunk(chunk_year)

    def _write_first_chunk(self, chunk: xr.Dataset):
        with netCDF4.Dataset(self.paths[-1], mode="w") as ds:
            ds.title = "Trajectories of Floating Marine Debris"
            ds.institution = "The Ocean Cleanup"
            ds.source = f"ADVECTOR Version {__version__}"

            config_group = ds.createGroup("configfile")
            with netCDF4.Dataset(self.configfile_path, mode="r") as configfile:
                copy_dataset(configfile, config_group)

            ds.createDimension("p_id", len(chunk.p_id))
            ds.createDimension("time", None)  # unlimited dimension

            # Variables that don't change between chunks
            p_id = ds.createVariable("p_id", chunk.p_id.dtype, ("p_id",))
            p_id[:] = chunk.p_id.values

            release_date = ds.createVariable("release_date", np.float64, ("p_id",))
            release_date.units = "seconds since 1970-01-01 00:00:00.0"
            release_date.calendar = "gregorian"
            release_date[:] = chunk.release_date.values.astype("datetime64[s]").astype(np.float64)

            radius = ds.createVariable("radius", np.float64, ("p_id",))
            radius.units = "meters"
            radius[:] = chunk.radius.values.astype(np.float64)

            density = ds.createVariable("density", np.float64, ("p_id",))
            density.units = "kg m^-3"
            density[:] = chunk.density.values.astype(np.float64)

            # Variables that expand/change between chunks
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

            depth = ds.createVariable("depth", chunk.depth.dtype, ("p_id", "time"))
            depth.units = "meters"
            depth.positive = "up"
            depth[:] = chunk.depth.values

            exit_code = ds.createVariable("exit_code", np.byte, ("p_id",))
            exit_code.description = "These codes are returned by the kernel when unexpected behavior occurs and the" \
                                    "kernel must be terminated.  Their semantic meaning is provided in the " \
                                    "'code_to_meaning' attribute of this variable."
            exit_code.code_to_meaning = str({code: meaning for code, meaning in EXIT_CODES.items() if code >= 0})
            exit_code[:] = chunk.exit_code.values

    def _append_chunk(self, chunk: xr.Dataset):
        with netCDF4.Dataset(self.paths[-1], mode="a") as ds:
            time = ds.variables["time"]
            start_t = len(time)
            time[start_t:] = chunk.time.values.astype("datetime64[s]").astype(np.float64)

            lon = ds.variables["lon"]
            lon[:, start_t:] = chunk.lon.values

            lat = ds.variables["lat"]
            lat[:, start_t:] = chunk.lat.values

            depth = ds.variables['depth']
            depth[:, start_t:] = chunk.depth.values

            exit_code = ds.variables["exit_code"]
            # overwrite with most recent codes; by design, nonzero codes cannot change
            exit_code[:] = chunk.exit_code.values


def copy_dataset(source: netCDF4.Dataset, destination: netCDF4.Dataset):
    """adapted from https://stackoverflow.com/a/49592545"""
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
