# Sourcefile Specifications
Source files contain the initial state of particles, and are encoded as netcdf datasets.  There is some difference between the sourcefiles required by the 2D and 3D Kernels.  In general, the 3D sourcefiles are the same as 2D, but with some extra variables.

## 2D Sourcefiles

### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| p_id | integer | indexes particles |

### Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| p_id | (p_id) | integer | numeric id of particle (coordinate variable) |
| lon | (p_id) | numeric | initial longitude of particle, degrees E, domain [-180, 180) |
| lat | (p_id) | numeric | initial latitude of particle, degrees N, domain [-90, 90] |
| release_date | (p_id) | datetime (CF- or ISO-compliant) | timestamp at which particle enters simulation |

## 3D Sourcefiles
These are the same as 2D sourcefiles, with some extra required variables:
### Additional Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| depth | (p_id) | numeric | initial depth of particle, m, positive up |
| radius | (p_id) | numeric | nominal radius of particle (m), i.e. radius of a sphere with same volume as particle |
| density | (p_id) | numeric | density of particle (kg m^-3) |
| corey_shape_factor | (p_id) | numeric | Represents the shape of the particle.  Defined as c/sqrt(a*b), where a, b, and c are the longest, intermediate, and shortest perpendicular dimensions of the particle. |

## Notes
* Additional variables beyond those specified may be present in sourcefiles.  If they have only one dimenion, `(p_id)`, they will be copied to the root group of outputfiles unchanged; otherwise they will be discarded.  This feature can be useful to add additional silent information to particles, such as their country of origin.