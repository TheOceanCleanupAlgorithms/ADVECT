## Sourcefile Specifications
Source files contain the initial state of particles, and are encoded as netcdf datasets.  They adhere to the following dataset structure:

#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| p_id | integer | indexes particles |

#### Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| p_id | (p_id) | integer | numeric id of particle (coordinate variable) |
| lon | (p_id) | numeric | initial longitude of particle, degrees E, domain [-180, 180) |
| lat | (p_id) | numeric | initial latitude of particle, degrees N, domain [-90, 90] |
| depth | (p_id) | numeric | initial depth of particle, m, positive up |
| radius | (p_id) | numeric | nominal radius of particle (m), i.e. radius of a sphere with same volume as particle |
| density | (p_id) | numeric | density of particle (kg m^-3) |
| corey_shape_factor | (p_id) | numeric | Represents the shape of the particle.  Defined as c/sqrt(a*b), where a, b, and c are the longest, intermediate, and shortest perpendicular dimensions of the particle. |
| release_date | (p_id) | datetime (CF- or ISO-compliant) | timestamp at which particle enters simulation |
