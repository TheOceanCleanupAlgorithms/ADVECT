# Data Specifications

## Sourcefiles
Source files contain the initial state of particles, and are encoded as netcdf datasets.  Two formats of sourcefile are accepted: advector (default, native), and trashtracker (legacy support).

### Advector Sourcefiles
Advector sourcefiles adhere to the following dataset structure:

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

### TrashTracker Sourcefiles
Trashtracker sourcefiles adhere to the following dataset structure:

#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| x | integer | indexes particles |

#### Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| id | (x) | integer | numeric id of particle |
| lon | (x) | numeric | initial longitude of particle, degrees E, any domain (will be coerced to [-180, 180) |
| lat | (x) | numeric | initial latitude of particle, degrees N, domain [-90, 90] |
| releaseDate | (x) | numeric, Matlab 'datenum' representation | timestamp at which particle enters simulation |
| unsd | (x) | integer | UNSD M49 country code.  Currently ignored. |
## Outputfiles
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

#### Exit Codes

| Code | Semantic Name | Description |
| --- | --- | --- |
| 0 | SUCCESS | nothing out of the ordinary |
| 1 | NULL_LOCATION | particle position became NaN during kernel execution |
| 2 | INVALID_LATITUDE | particle latitude outside of [-90, 90] |

## Forcing Data
### Sea Surface Current
Encoded as a netcdf dataset, and must adhere to the following specifications:

#### Dimensions
| Name | Data Type | Description |
| --- | --- | --- |
| lon | integer | indexes longitude |
| lat | integer | indexes latitude |
| time | integer | indexes time |
| depth | integer | indexes depth |

#### Variables
| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| lon | (lon) | numeric | longitude of grid cell centers, degrees E, domain [-180, 180) (coordinate variable).  Values must be sorted and equally spaced. |
| lat | (lat) | numeric | latitude of grid cell centers, degrees E, domain [-90, 90] (coordinate variable).  Values must be sorted and equally spaced. |
| depth | (depth) | numeric | depth of grid cell centers, meters, positive up (coordinate variable).  Values must be sorted but can have arbitrary spacing. |
| time | (time) | datetime (CF- or ISO-compliant) | time coordinate of field (coordinate variable). Values must be sorted and equally spaced. |
| U | (time, depth, lat, lon) | numeric | zonal component of velocity (m/s), positive east.
| V | (time, depth, lat, lon) | numeric | meridional component of velocity (m/s), positive north. |
| W | (time, depth, lat, lon) | numeric | vertical component of velocity (m/s), positive up. |

### 10m Wind
Encoded as a netcdf dataset.  Note that the near-surface, turbulent wind-mixing algorithm assumes that each wind data is an average over a timeframe of ~1 day, as it assumes the sea state is fully developed based on the closest wind observation.  If you pass in wind data with sub-daily intervals along the time dimension, the wind mixing will become less realistic.

Must adhere to the following specifications:

#### Dimensions
| Name | Data Type | Description |
| --- | --- | --- |
| lon | integer | indexes longitude |
| lat | integer | indexes latitude |
| time | integer | indexes time |

#### Variables
| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| lon | (lon) | numeric | longitude of grid cell centers, degrees E, domain [-180, 180) (coordinate variable).  Values must be sorted and equally spaced. |
| lat | (lat) | numeric | latitude of grid cell centers, degrees E, domain [-90, 90] (coordinate variable).  Values must be sorted and equally spaced. |
| time | (time) | datetime (CF- or ISO-compliant) | time coordinate of field (coordinate variable). Values must be sorted and equally spaced. |
| U | (lon, lat, time) | numeric | zonal component of velocity (m/s), positive east.
| V | (lon, lat, time) | numeric | meridional component of velocity (m/s), positive north.
Notes:
* Non-standard names on any dimension or variable are supported via name mappings passed into `run_advector` in src/run_advector.py.
* Any "packed" variables must have the attributes `scale_factor`, `add_offset`, and `_FillValue` in order to be properly unpacked.  See the [netcdf conventions](https://www.unidata.ucar.edu/software/netcdf/docs/attribute_conventions.html) for more details.
* Domain [0, 360) is supported for longitude but the conversion to [-180, 180) is computationally expensive.
