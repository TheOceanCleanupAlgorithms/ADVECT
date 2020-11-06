# Data Specifications
## Sourcefiles
Source files contain the initial state of particles, and are encoded as netcdf datasets.  Two formats of sourcefile are accepted: advector (default, native), and trashtracker (legacy support).
### Advector Sourcefiles
Advector sourcefiles adhere to the following dataset structure:

Dimensions

| Name | Type | Description |
| --- | --- | --- |
| id | integer | indexes particles |

Variables

| Name | Dimension | Type | Description |
| --- | --- | --- | --- |
| id | id | integer | numeric id of particle (coordinate variable) |
| lon | id | numeric | initial longitude of particle, degrees E, domain [-180, 180) |
| lat | id | numeric | initial latitude of particle, degrees N, domain [-90, 90] |
| release_date | id | datetime (CF- or ISO-compliant) | timestamp at which particle enters simulation |
Note: other variables may be present, they will likely be ignored.  Undefined behavior.
### TrashTracker Sourcefiles
Trashtracker sourcefiles adhere to the following dataset structure:

Dimensions

| Name | Type | Description |
| --- | --- | --- |
| x | integer | indexes particles |

Variables

| Name | Dimension | Type | Description |
| --- | --- | --- | --- |
| id | x | integer | numeric id of particle |
| lon | x | numeric | initial longitude of particle, degrees E, any domain (will be coerced to [-180, 180) |
| lat | x | numeric | initial latitude of particle, degrees N, domain [-90, 90] |
| releaseDate | x | numeric, Matlab 'datenum' representation | timestamp at which particle enters simulation |
| unsd | x | integer | UNSD M49 country code.  Currently ignored. |
## Outputfiles
Outputfiles contain the state of particles along some timeseries.  They are essentially Advector-format sourcefiles with an additional time dimension.  They adhere to the following specifications:

Dimensions

| Name | Type | Description |
| --- | --- | --- |
| p_id | integer | indexes particles |
| time | integer | indexes timeseries |

Variables

| Name | Dimension | Type | Description |
| --- | --- | --- | --- |
| p_id | p_id | int64 | numeric id of particle (coordinate variable) |
| time | time | int64 (CF-compliant datetime format) | timeseries along which particle states have been saved (coordinate variable) |
| lon | (p_id, time) | float32 | longitude of particle, degrees E, domain [-180, 180) |
| lat | (p_id, time) | float32 | latitude of particle, degrees N, domain [-90, 90] |
| release_date | (time, p_id) | int64 (CF-compliant datetime format) | timestamp at which particle entered simulation |

## Forcing Data


### 2D Vector Fields (Sea Surface Current, Surface Wind)
