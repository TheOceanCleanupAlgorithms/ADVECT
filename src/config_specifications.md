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


## Seawater Density Profile
The density of seawater in the real ocean varies across space and time.  Variation in time and the horizontal dimensions, however, is small (~10 kg m^-3) compared to the density differential between commercial plastics and seawater (~50-500 kg m^-3).  The variation in depth (challenger deep vs surface ≈ 50 kg m^-3) is much larger, as seawater is only _mostly_ incompressible.  Thus for the sake of simplicity, ADVECTOR assumes that the density of seawater only varies with depth; this depth dependent density profile is encoded in the following variables within the configfile:

#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| z_sd | integer | indexes variable z_sd |

#### Variables
| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| z_sd | (z_sd) | numeric | depth coordinate for seawater_density (m, positive up) (coordinate variable) |
| seawater_density | (z_sd) | numeric | density of seawater (kg m^-3) |
