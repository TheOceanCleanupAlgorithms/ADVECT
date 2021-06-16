# ADVECTOR V1.0
The ADVECTOR is a whole-ocean marine debris transport model which is built to handle high throughput and massive datasets.  It models the transport of debris based on its size, shape, and density, and simulates basic physical processes including 3D ocean current-driven advection, wind-driven drift, wind-driven near-surface vertical mixing, buoyancy-driven vertical transport, and eddy-driven diffusion.  It automatically processes forcing datasets arbitrarily larger than memory capacity, and supports fully parallelized computation on CPUs and GPUs via OpenCL.

## Model Description
The ADVECTOR contains solvers (kernels) for two domains: ocean surface, and whole-ocean.
* 2D kernel: model domain is constrained to the surface of the ocean (assumption: floating debris), and debris particles are idealized, with no consideration of their size/shape/density.
* 3D kernel: model domain is the whole oceans, from surface to bathymetry, and physical processes depend on the size/shape/density of debris.

### 2D Kernel
Simulations using the 2D kernel can be run via `src/run_advector_2D.py`.  Particles are released at some location in space and time, and are transported according to the following physical processes:
#### Surface ocean current-driven advection
Particles are transported in a time-evolving 2D velocity field of surface ocean currents, which the user must provide.  The particles are advected according to one of two schemes: forward-Euler, or a second-order Taylor-expansion scheme which corrects for the outward-drift error the Euler method experiences in a curved field.
#### Wind-driven drift
Optionally, the user may provide a time-evolving 2D velocity field of 10-meter wind, which will move particles according to a user-provided windage coefficient.
#### Eddy-driven diffusion
Finally, the user may specify a constant eddy diffusivity, which will add random noise to the particle's movements.  This simulates the effect of eddies smaller than the spatial resolution of the ocean currents.
#### Boundary processes
The model domain only includes the surface waters of the ocean (as defined by the non-null region in the ocean current vectorfield); particles cannot leave this domain, and thus the model does not include beaching.  Instead, when particles are pushed against a coastline, their onshore displacement component is cropped to keep them in the model domain, generally resulting in a lateral displacement, as if the boundary was frictionless.

### 3D Kernel
Simulations using the 3D kernel can be run via `src/run_advector_3D.py`.  Each particle is initialized with a size, shape, density, release date, and release location.  Upon release, particles are transported according to the following physical processes:
#### 3D ocean current-driven advection
Particles are transported according to a time-evolving 3D velocity field of ocean currents, which the user provides.  The particles are advected according to one of two schemes: forward-Euler, or a 3D adaptation of the second-order Taylor-expansion scheme from the 2D kernel.
#### Buoyancy-driven transport
The user must provide a time-evolving 3D dataset containing the density of seawater in the ocean domain.  Particles are transported vertically according to their terminal sinking (or rising) velocity, which is calculated using their size, shape, and density, as well as the density of the surrounding seawater.
#### Wind-driven drift
Optionally, the user may provide a time-evolving 2D velocity field of 10-meter wind, which will move particles floating at the surface based on a parameterization which depends on their emerged surface area.  The user may optionally provide a multiplier which scales this drift, for the sake of experimentation.
#### Wind-driven vertical mixing
If wind is provided, the user may optionally enable the simulation of wind-driven vertical mixing.  Mixing transport is based on an equilibrium between a particle's rising velocity and the size of ocean waves (estimated from wind).
#### Eddy-driven diffusion
Finally, the user may specify a vertical profile of vertical and horizontal eddy diffusivities, which will add noise to the particle's movements according to its depth.  This simulates eddies smaller than the spatial resolution of the ocean currents, and allows the user the flexibility to account for the depth-dependent nature of eddy diffusivity in the world's oceans.
#### Boundary processes
The model domain only includes the waters of the ocean above bathymetry (as defined by the non-null region in the ocean current vectorfield); particles cannot leave this domain, and thus the model does not include beaching or sedimentation.  Instead, when particles are pushed against coastline/bathymetry, their out-of-domain displacement components are cropped to keep them in the model domain.  This is the 3D analog of the frictionless coastlines used in the 2D kernel, and similarly allows particles to travel parallel to domain boundaries.

## Installation Instructions
1. In a terminal, clone this repository and navigate to its root.
2. Install ADVECTOR as a package
    ```
   pip install -e .
   ```
3. (Optional) Run tests

    To ensure everything is working before you go through the effort of downloading forcing data, run `python -m pytest` from the project root.  If any tests do not pass, a first step is to check out the "hardware compatability" section below.
5. Acquire forcing data

    Instructions for downloading sample forcing data can be found in `examples/README.txt`.
    
    At minimum, ADVECTOR's 2D kernel requires surface current data, and the 3D kernel requires whole-ocean current, as well as seawater density.  Often, ocean models only offer their zonal/meridional current velocity to download; ADVECTOR comes bundled with a tool called the INTEGRATOR which can generate vertical velocity fields from zonal/meridional velocity fields, using the continuity equation.  Check out `INTEGRATOR/README.md` for more information.

6. Run example advection

    Once you're all set up with the necessary forcing data according to `examples/README.txt`, you can execute `examples/ECCO_advect_2D.py` and/or `examples/ECCO_advect_3D.py`. 

### Hardware compatability
At this time, ADVECTOR only has known support for CPUs/GPUs with opencl driver versions 1.1, 1.2, and 2.1.  Running tests is one way to check if your hardware is compatible.  If they fail, you can run this in a python prompt to directly check your driver version:
   ```
   import pyopencl
   print(pyopencl.create_some_context(interactive=True).devices[0].driver_version)
   ```
   Follow the instructions to select a compute device, and its driver version will be displayed.
