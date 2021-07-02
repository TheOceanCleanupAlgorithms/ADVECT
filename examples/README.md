## OVERVIEW
This document walks you through downloading one month of forcing data to test out ADVECTOR.

### 1. Acquire Forcing Data.
Simply run `python examples/download_example_data.py` and provide a path to the directory where the data will go.  The currents and seawater density come from JPL's ECCO model, which provides a high quality coarse-resolution ocean reanalysis.  You'll need to create a NASA earthdata account to authenticate the download; the script provides a link.  The wind data comes from the NCEP-NCAR Reanalysis 2, which is a bit outdated, but hosts easily downloadable 10m wind data.


### 2. Run the example scripts
Once you have your forcing data, just run `python examples/ECCO_advect_2D.py` and `python examples/ECCO_advect_3D.py` to see ADVECTOR in action!  You'll receive some prompts, asking where the forcing data is (same path you put into step 1), where to put the model output, and what compute device to use for the computation.  Follow the instructions and you're off to the races!
