# Configfile Specifications

The 3D Kernel requires enough physical configuration to justify an analog to a sourcefile, called a **configfile**.  Configfiles encode physical parameterizations by means of a netcdf dataset.  They contain vertical profiles of eddy diffusivity, as well as scalar physics parameters.  This document details the required netcdf dataset structure within a configfile.

## Eddy Diffusivity Profiles
Transport due to eddies smaller than the resolution of the current files is paramaterized through a random walk; the amplitude of this random walk depends on eddy diffusivity.  In the real ocean, eddy diffusivity varies in time and space, and has different properties in the horizontal vs vertical.  The 3D Kernel does distinguish between horizontal and vertical diffusivity, but makes the simplifying assumption that these diffusivities vary only with depth.

### Horizontal Eddy Diffusivity
Horizontal eddy diffusivity parameterizes a particle's random walk in latitude/longitude.
#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| z_hd | integer | indexes variable z_hd |

#### Variables
| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| z_hd | (z_hd) | numeric | depth coordinate for horizontal_diffusivity (m, positive up) (coordinate variable) |
| horizontal_diffusivity | (z_hd) | numeric | horizontal eddy diffusivity (m^2 s^-1) |

### Vertical Eddy Diffusivity
Vertical eddy diffusivity parameterizes a particle's random walk in depth.
#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| z_vd | integer | indexes variable z_vd |

#### Variables
| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| z_vd | (z_vd) | numeric | depth coordinate for vertical_diffusivity (m, positive up) (coordinate variable) |
| vertical_diffusivity | (z_vd) | numeric | vertical eddy diffusivity (m^2 s^-1) |


### Wind Mixing Constants
The 3D Kernel's parameterization of wind mixing relies on a couple of physical constraints.  These are optional in the configfile, and they will be set to default values if not present.

#### Scalar Variables
It's recommended to leave these at their default values (you can do this by omitting them) unless you have a good reason to do otherwise.  These variables dramatically impact the wind-mixing component of the model.

| Name | Data Type | Default Value | Description |
| --- | --- | --- |--- |
| max_wave_height | numeric | 20 | meters; maximum physically reasonable significant wave height; at very high wind speeds the formula which calculates these produces non-physical values, so a cap must be enforced.  The default 20 comes from the approximate world record as of 2020 (https://wmo.asu.edu/content/World-Highest-Wave-Buoy) |
| wave_mixing_depth_factor | numeric | 10 | unitless; controls the maximum depth a particle can be mixed to by wave action, governed by max_depth = -(WAVE_MIXING_DEPTH_MULTIPLIER * significant_wave_height).  In the adopted wave-mixing model, the probability of mixing to a depth z decreases with depth, but is never zero.  This will occasionally result in nonphysical mixing to extreme depths.  Thus, the probability must be set to zero at some depth.  The natural formulation for this depth is as a scalar multiple of the wave height.  The default value of 10 is chosen based on results in moderate wind conditions from D'Asaro et al (2013) Figure 1.  Paired with the default max wave height of 20m, the default multiple produces a maximum mixing depth of 200m under the most extreme wind conditions, which is similar to the mixed layer depth of the Wintertime North Atlantic/North Pacific and of the Southern Ocean.
