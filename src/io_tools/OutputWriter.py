from pathlib import Path
import xarray as xr
import netCDF4
import numpy as np


class OutputWriter:
    def __init__(self, out_dir: Path):
        if not out_dir.is_dir():
            out_dir.mkdir()

        self.folder_path = out_dir
        self.current_year = None
        self.paths = []

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
            ds.createDimension("time", None)  # unlimited dimension
            ds.createDimension("p_id", len(chunk.p_id))

            time = ds.createVariable("time", np.float64, ("time",))
            time.units = "seconds since 1970-01-01 00:00:00.0"
            time.calendar = "gregorian"
            time[:] = chunk.time.values.astype("datetime64[s]").astype(np.float64)

            p_id = ds.createVariable("p_id", chunk.p_id.dtype, ("p_id",))
            p_id[:] = chunk.p_id.values

            lon = ds.createVariable("lon", chunk.lon.dtype, ("p_id", "time"))
            lon.units = "Degrees East"
            lon[:] = chunk.lon.values

            lat = ds.createVariable("lat", chunk.lat.dtype, ("p_id", "time"))
            lat.units = "Degrees North"
            lat[:] = chunk.lat.values

            release_date = ds.createVariable("release_date", np.float64, ("p_id",))
            release_date.units = "seconds since 1970-01-01 00:00:00.0"
            release_date.calendar = "gregorian"
            release_date[:] = chunk.release_date.values.astype("datetime64[s]").astype(np.float64)

            exit_code = ds.createVariable("exit_code", np.ubyte, ("p_id",))
            for meaning, code in {'SUCCESS': 0, 'FAILURE': 1, 'INVALID_ADVECTION_SCHEME': 2, 'NULL_LOCATION': 3}.items():
                setattr(exit_code, str(code), meaning)
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

            exit_code = ds.variables["exit_code"]
            # overwrite with most recent codes; by design, nonzero codes cannot change
            exit_code[:] = chunk.exit_code.values
