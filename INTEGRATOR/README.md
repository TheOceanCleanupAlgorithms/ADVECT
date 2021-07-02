# The INTEGRATOR
This is a tool which calculates an oceanic vertical velocity field (W), given zonal and meridional velocity fields (U and V).  ADVECTOR in 3D requires the full 3D current field; however, most publicly available model outputs do not include W.  As such, we have included the INTEGRATOR for your convenience.

## File Descriptions

* `integrator.py` contains the key function which, given an xarray dataset containing U and V (and coordinates), will generate W for you.  It is fully dask compatible, so you can chunk your computation along depth levels if dealing with very hefty files.

* `generate_ECCO_W.py` is a script which, provided you have the right files (or modify the paths), demonstrates usage, and plots a comparison between the W calculated from ECCO U/V with the W provided by ECCO directly.  The high correspondence is a good confirmation of the quality of the method.

* `w_integration_methodology.ipynb` contains a detailed description of the mathematical method used to integrate the continuity equation.

* `seawater_denity_profile.nc` holds a profile of seawater density versus depth for use in `integrator.py`.  Values calculated from 2015 monthly Temp/Salinity data from HYCOM.
