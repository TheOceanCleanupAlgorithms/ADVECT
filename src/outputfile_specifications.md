## Outputfile Specifications
Outputfiles contain the particle trajectories computed by the model, as well as metadata which describe in detail the configuration of ADVECTOR which produced them.  They adhere to the following specifications:

#### Groups
NetCDF-4 added support for hierarchical groups within netCDF datasets; each group acts as its own dataset.  We use this functionality to include a bunch of metadata along with the trajectories, without cluttering the trajectory dataset.

| Name | Description |
| --- | --- |
| root group | contains trajectory data and particle information, as well as descriptive attributes |
| configfile | a copy of the configfile passed to ADVECTOR |
| sourcefile | a copy of the sourcefile passed to ADVECTOR |
| currents_meta | contains the coordinates of the current dataset passed to ADVECTOR, as well as the global attributes from the first zonal current file |
| wind_meta | contains the coordinates of the wind dataset passed to ADVECTOR, as well as the global attributes from the first zonal wind file |

We will only give detailed specifications on the contents of the root group, as this is the primary data product; the others exist for the sake of fully describing the model configuration which produced the outputfile.

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
| depth | (p_id, time) | float32 | depth of particle, meters, positive up |
| release_date | (p_id) | float64 (CF-compliant datetime format; epoch seconds) | timestamp after which particle entered simulation |
| radius | (p_id) | float64 | nominal radius of particle, meters (see sourcefile specs for more info) |
| density | (p_id) | float64 | density of particle, kg m^-3 |
| exit_code | (p_id) | uint8 | relays information about any non-fatal errors encountered during kernel execution. See below. |

Additional variables with dimensions `(p_id)` may be present, copied from the sourcefile.  See sourcefile_specifications.md.

#### Exit Codes

| Code | Semantic Name | Description |
| --- | --- | --- |
| 0 | SUCCESS | nothing out of the ordinary |
| 1 | NULL_LOCATION | particle position became NaN during kernel execution |
| 2 | INVALID_LATITUDE | particle latitude outside of [-90, 90] |
