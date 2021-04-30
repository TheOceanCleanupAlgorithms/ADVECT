# Outputfile Specifications
Outputfiles contain the particle trajectories computed by the model, as well as metadata which describe in detail the configuration of ADVECTOR which produced them.  There is some difference between the outputfiles produced by the 2D and 3D Kernels.

## 2D Outputfiles

### Groups
NetCDF-4 added support for hierarchical groups within netCDF datasets; each group acts as its own dataset.  We use this functionality to include a bunch of metadata along with the trajectories, without cluttering the trajectory dataset.

| Name | Description |
| --- | --- |
| root group | contains trajectory data and particle information, as well as descriptive attributes |
| configfile | a copy of the configfile passed to ADVECTOR (only present for 3D Outputfiles) |
| sourcefile | a copy of the sourcefile passed to ADVECTOR |
| currents_meta | contains the coordinates of the current dataset passed to ADVECTOR, as well as the global attributes from the first file in the dataset |
| wind_meta | contains the coordinates of the wind dataset passed to ADVECTOR, as well as the global attributes from the first file in the dataset |

We will only give detailed specifications on the contents of the root group, as this is the primary data product; the others exist for the sake of fully describing the model configuration.

### Root Group

#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| p_id | integer | indexes particles |
| time | integer | indexes timeseries |

#### Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| p_id | (p_id) | int64 | numeric id of particle (coordinate variable) |
| time | (time) | float64 (CF-compliant datetime format; epoch seconds) | timeseries along which particle states have been saved (coordinate variable) |
| lon | (p_id, time) | float32 | longitude of particle, degrees E, domain [-180, 180) |
| lat | (p_id, time) | float32 | latitude of particle, degrees N, domain [-90, 90] |
| release_date | (p_id) | float64 (CF-compliant datetime format; epoch seconds) | timestamp after which particle entered simulation |
| exit_code | (p_id) | int8 | relays information about any non-fatal errors encountered during kernel execution. Defined at end of document. |

Additional variables with dimensions `(p_id)` may be present, copied from the sourcefile.  See sourcefile_specifications.md.

## 3D Outputfiles
These are generally identical to 2D Outputfiles, just with additional groups/variables.

### Additional Groups
| Name | Description |
| --- | --- |
| configfile | a copy of the configfile passed to ADVECTOR (only present for 3D Outputfiles) |
| seawater_density_meta | contains the coordinates of the seawater density dataset passed to ADVECTOR, as well as the global attributes from the first file |

### Root Group

#### Additional Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| depth | (p_id, time) | float32 | depth of particle, meters, positive up |
| radius | (p_id) | float64 | "nominal radius" of particle, meters (see sourcefile_specifications.md for definition) |
| density | (p_id) | float64 | density of particle, kg m^-3 |
| corey_shape_factor | (p_id) | float64 | "corey shape factor" of particle, unitless; see sourcefile_specifications.md for definition |

## Exit Codes

| Code | Semantic Name | Description |
| --- | --- | --- |
| 0 | SUCCESS | nothing out of the ordinary |
| 1 | NULL_LOCATION | particle position became NaN during kernel execution |
| 2 | INVALID_LATITUDE | particle latitude outside of [-90, 90] |
| 3 | SEAWATER_DENSITY_LOOKUP_FAILURE | sparse search for nearby seawater density data returned no results |
