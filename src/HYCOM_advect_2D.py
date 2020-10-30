"""
advect on ECCO surface currents
"""

from kernel_wrappers.Kernel2D import AdvectionScheme
from run_advector import run_advector
from plotting.plot_advection import plot_ocean_advection
from tools.open_sourcefile import SourceFileType


EDDY_DIFFUSIVITY = 0  # m^2 / s
''' Sylvia Cole et al 2015: diffusivity calculated at a 300km eddy scale, global average in top 1000m, Argo float data.
  This paper shows 2 orders of magnitude variation regionally, not resolving regional differences is a big error source.
  Additionally, the assumption here is that 300km eddies are not resolved by the velocity field itself.  If they are,
  we're doubling up on the eddy transport.  For reference, to resolve 300km eddies, the grid scale probably needs to be
  on order 30km, which at the equator would be ~1/3 degree.
'''


if __name__ == '__main__':
    out_path = run_advector(
        outputfile_path='../outputfiles/1993_HYCOM.nc',
        sourcefile_path='V:/SourcesForTest/Source_1_1993/outputfolder/parts_source_1993_c.nc',
        uwater_path='E:/CURRENT/u/u_1993*.nc',
        vwater_path='E:/CURRENT/v/v_1993*.nc',
        advection_start='1993-01-01T01',
        timestep_seconds=3600,
        num_timesteps=24*365,
        save_period=24,
        advection_scheme=AdvectionScheme.taylor2,
        eddy_diffusivity=EDDY_DIFFUSIVITY,
        platform_and_device=(0,),
        sourcefile_varname_map={'releaseDate': 'release_date'},
        currents_varname_map={'water_u': 'U', 'water_v': 'V'},
        verbose=True,
        source_file_type=SourceFileType.old_source_files,
        memory_utilization=.01,
    )

    plot_ocean_advection(out_path)
