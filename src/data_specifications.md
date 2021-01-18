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
| release_date | (p_id) | datetime (CF- or ISO-compliant) | timestamp at which particle enters simulation |
Note: other variables may be present, they will likely be ignored.  Undefined behavior.
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
Outputfiles contain the state of particles along some timeseries.  They are essentially Advector-format sourcefiles with an additional time dimension.  They adhere to the following specifications:

#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| p_id | integer | indexes particles |
| time | integer | indexes timeseries |

#### Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| p_id | (p_id) | int64 | numeric id of particle (coordinate variable) |
| time | (time) | int64 (CF-compliant datetime format) | timeseries along which particle states have been saved (coordinate variable) |
| lon | (p_id, time) | float32 | longitude of particle, degrees E, domain [-180, 180) |
| lat | (p_id, time) | float32 | latitude of particle, degrees N, domain [-90, 90] |
| release_date | (p_id) | int64 (CF-compliant datetime format) | timestamp after which particle entered simulation |
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
Encoded as a netcdf dataset, and must adhere to the following specifications:

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
