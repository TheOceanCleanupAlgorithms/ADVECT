import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from interp_ECCO_to_latlon import interpolate_variable


def interp_currents(
    native_dir: Path,
    out_dir: Path,
    grid_path: Path,
    resolution_deg: float,
):
    variables = {
        "EVEL": "U",  # ECCO_native varname: [local varname, vertical grid name]
        "NVEL": "V",
        "WVELMASS": "W",
    }
    for ECCO_varname, local_varname in variables.items():
        new_grid_delta_lat = resolution_deg
        new_grid_delta_lon = resolution_deg
        new_grid_min_lat = (
            -90 + new_grid_delta_lat / 2
        )  # domain of interpolated field (deg)
        new_grid_max_lat = 90 - new_grid_delta_lat / 2
        new_grid_min_lon = -180 + new_grid_delta_lon / 2
        new_grid_max_lon = 180 - new_grid_delta_lon / 2
        interpolate_variable(
            ECCO_varname,
            local_varname,
            new_grid_min_lat,
            new_grid_max_lat,
            new_grid_delta_lat,
            new_grid_min_lon,
            new_grid_max_lon,
            new_grid_delta_lon,
            native_dir=native_dir,
            interp_dir=out_dir,
            native_grid_path=grid_path,
        )
