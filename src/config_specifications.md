# Configfile Specifications

Configfiles encode ADVECTOR's physical parameterizations by means of a netcdf dataset.  As of now, they just contain vertical profiles of eddy diffusivity.  In the future, they may also contain scalar physics parameters, or other paramaterizations.  This document details the required netcdf dataset structure within a configfile.


## Eddy Diffusivity Profiles
Transport due to eddies smaller than the resolution of the current files is paramaterized through a random walk; the amplitude of this random walk depends on eddy diffusivity.  In the real ocean, eddy diffusivity varies in time, space, and coordinate direction.  ADVECTOR makes the simplifying assumption that eddy diffusivity varies only with depth and coordinate direction.

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
