# ADVECTOR
The emerging interest in vertical dynamics and a constant trend towards increased spatial and temporal resolution necessitates a fully distributed, computationally efficient solution to the problem of modeling marine litter transport at sea.  As such, this software aims to be not just computationally efficient, but fully distributed as well, in order to take full advantage of massively parallel hardware architectures such as GPUs and CPU clusters.  The OpenCL paradigm accomplishes both of these goals.

## Features and Timeline
### V0: Sea Surface Advection
Version 0 aims to be a functionally equivalent to the TrashTracker model developed by Laurent Lebreton.  Intended for global dispersion studies, it will use a 2D, second-order advection scheme, and will include sea-surface-current advection, eddy diffusion, windage, and a "slippery" coastline handling, like trashtracker.
### V1: 3D Advection (Buoyancy Driven)
Version 1 will consider depth, and will require vertical current as an input.  It will use a 3D second-order advection scheme and will support buoyancy-driven vertical movement.
### V2: Elaborate Vertical Transport Mechanisms
Version 2 will add new vertical transport mechanisms, and will support trilinear field interpolation for the advection algorithm.
### V3: Boundary Processes (Coasts/Bathymetry)
Version 3 will expand the consideration of coastal processes beyond simple beaching, will consider processes at the seafloor, and may add support for new advection kernels.
## Setup
You can set up your conda environment with `conda env create -f environment.yml`. Get miniconda [here](https://docs.conda.io/en/latest/miniconda.html).

Depending on your use-case, you may need to create a .env file with the following keys:
```
copernicus_uname=<username>  # for downloading cmems data
copernicus_pwd=<password>
```
 