## OVERVIEW
This document walks you through downloading some sample data to test out ADVECTOR.  If you want to test the 2D version only, you need only to download ocean currents and wind data (optional).  If you want to test the 3D version, you'll additionally need to download seawater density data. 

### Ocean Current Data
JPL's ECCO model provides a convenient python API for downloading and processing their model output in coarse resolution.  Run `python examples/ECCO/download_and_process_ECCO.py` to acquire 3D ocean currents for the year 2015.

### 10-meter wind
The NCEP-NCAR Reanalysis 2, while a bit outdated, hosts easily downloadable 10m wind data.  Run `python examples/ncep_ncar_doe_ii/download_10m_wind.py` to acquire data for the year 2015.

### Seawater Density
ADVECTOR 3D requires seawater density.  ECCO also provides this; run `python examples/ECCO/download_and_process_density.py` to acquire data for the year 2015.

### Running the Scripts
Once you have your forcing data, just run `python examples/ECCO_advect_2D.py` or `python examples/ECCO_advect_3D.py`.  You'll receive some prompts, asking what device to use for the computation.  You can of course use your CPU, or if your computer has a dedicated GPU, try that out!
