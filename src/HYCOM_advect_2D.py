"""
advect on ECCO surface currents
"""

from kernel_wrappers.Kernel2D import AdvectionScheme
from run_advector import run_advector
from plotting.plot_advection import plot_ocean_advection
from tools.open_sourcefiles import SourceFileType
from datetime import datetime


EDDY_DIFFUSIVITY = 0  # m^2 / s
""" Sylvia Cole et al 2015: diffusivity calculated at a 300km eddy scale, global average in top 1000m, Argo float data.
  This paper shows 2 orders of magnitude variation regionally, not resolving regional differences is a big error source.
  Additionally, the assumption here is that 300km eddies are not resolved by the velocity field itself.  If they are,
  we're doubling up on the eddy transport.  For reference, to resolve 300km eddies, the grid scale probably needs to be
  on order 30km, which at the equator would be ~1/3 degree.
"""

ADVECTION_START = datetime(1993, 1, 1)
ADVECTION_END = datetime(1994, 1, 1)

storage_folder = "/scratch-shared/peytavin/"

if __name__ == "__main__":
    out_path = run_advector(
        outputfile_path=storage_folder + "output/1993_1994_HYCOM.nc",
        sourcefile_path=storage_folder + "sources/parts_source*.nc",
        u_water_path=storage_folder + "metocean/CURRENT/u/u_199*.nc",
        v_water_path=storage_folder + "metocean/CURRENT/v/v_199*.nc",
        advection_start=ADVECTION_START.isoformat(),
        timestep_seconds=3600,
        num_timesteps=24 * (ADVECTION_END - ADVECTION_START).days,
        save_period=24,
        advection_scheme=AdvectionScheme.taylor2,
        eddy_diffusivity=EDDY_DIFFUSIVITY,
        platform_and_device=(0,),
        sourcefile_varname_map={"release_date": "release_date", "x": "id"},
        currents_varname_map={"water_u": "U", "water_v": "V"},
        verbose=True,
        source_file_type=SourceFileType.old_source_files,
        memory_utilization=0.5,
    )

    plot_ocean_advection(out_path)
