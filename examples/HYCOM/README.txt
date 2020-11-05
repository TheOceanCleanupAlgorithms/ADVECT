Instructions: How to run ADVECTOR on HYCOM current data (and optionally, MERRA-2 wind)


HYCOM CURRENT DATA
------------------
Off-the-shelf, HYCOM data is compliant with ADVECTOR netcdf vector field specifications, aside from some simple variable
name mapping which is handled in HYCOM_advect_2D.  You can download some data by executing
`python examples/HYCOM/download_hycom_currents.py`, if you don't already have access to HYCOM data.

Once you have some data, you may need to change around some paths in HYCOM_advect_2D.py, this should be pretty clear.

Note for TOC personnel:
In order to have well-defined scope, ADVECTOR requires that its input data adheres to netcdf conventions.  However,
many of the HYCOM datasets existing internally at TOC have lost important attributes which convention specifies must
exist for "packed" variables, due to a concatenation script in the trashtracker repo.  As such,
HYCOM data which has lost these attributes needs to have them added again.  The script "fix_trashtracker_hycom.sh"
will add these attributes back to the water_u and water_v variables, and must be run on any HYCOM dataset missing these
attributes before the files are usable by ADVECTOR.  See the script for usage instructions.


MERRA-2 WIND DATA
-----------------
Look in the script "download_merra2_wind.sh".  Follow directions; you need an earthdata account,
and need to run some terminal commands in order to set up authentication with this account.

SOURCEFILE
----------
Sourcefiles hold the initial state of the particles, and HYCOM_advect_2D.py needs a path to one.  Generate a uniform one
using `python examples/sourcefiles/generate_sourcefile.py -n <number of particles>`, or use your
own.  Note: if using one which was made for trashtracker, you need to make some changes to the arguments in
HYCOM_advect_2D.py; there are comments which will guide you.

RUNNING THE SCRIPT
------------------
Once you have your current data and your sourcefile (and optionally, wind data), and HYCOM_advect_2D.py has the correct
paths to these datasets, just run `python examples/HYCOM/HYCOM_advect_2D.py`.  You'll receive some prompts, asking what
device to use for the computation.  You can of course use your CPU, or if your computer has a dedicated GPU,
try that out!
