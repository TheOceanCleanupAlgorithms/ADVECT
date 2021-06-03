## OVERVIEW
This document walks you through downloading some sample data to test out ADVECTOR.

### Ocean Current / Seawater Density Data
JPL's ECCO model provides a high quality coarse-resolution ocean reanalysis.  Run `python examples/ECCO/download_and_process_ECCO_data.py` to acquire 3D ocean currents for the year 2015 (~25 GB on native grid, ~15 GB after interpolation), as well as monthly seawater density for the same timeframe (~1 GB).  You'll need to create a NASA earthdata account; the script provides a link.

### 10-meter wind
The NCEP-NCAR Reanalysis 2, while a bit outdated, hosts easily downloadable 10m wind data.  Run `python examples/ncep_ncar_doe_ii/download_10m_wind.py` to acquire data for the year 2015 (~100 MB)


### Running the Scripts
Once you have your forcing data, just run `python examples/ECCO_advect_2D.py` or `python examples/ECCO_advect_3D.py`.  You'll receive some prompts, asking what compute device to use for the computation.  Follow the instructions and you're off to the races!
