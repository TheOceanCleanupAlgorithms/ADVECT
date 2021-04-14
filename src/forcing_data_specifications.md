## Forcing Data Specifications

ADVECTOR requires three forcing datasets: Ocean Current Velocity, 10m Wind Velocity, and Seawater Density.

### Ocean Current Velocity
Encoded as a netcdf dataset, and must adhere to the following specifications:

#### Dimensions
| Name | Data Type | Description |
| --- | --- | --- |
| lon | integer | indexes longitude |
| lat | integer | indexes latitude |
| depth | integer | indexes depth |
| time | integer | indexes time |

#### Variables
| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| lon | (lon) | numeric | longitude of grid cell centers, degrees E, domain [-180, 180) (coordinate variable).  Values must be sorted and equally spaced. |
| lat | (lat) | numeric | latitude of grid cell centers, degrees E, domain [-90, 90] (coordinate variable).  Values must be sorted and equally spaced. |
| depth | (depth) | numeric | depth of grid cell centers, meters, positive up (coordinate variable).  Values must be sorted but can have arbitrary spacing. |
| time | (time) | datetime (CF- or ISO-compliant) | time coordinate of field (coordinate variable). Values must be sorted and ascending. |
| U | (time, depth, lat, lon) | numeric | zonal component of velocity (m/s), positive east.
| V | (time, depth, lat, lon) | numeric | meridional component of velocity (m/s), positive north. |
| W | (time, depth, lat, lon) | numeric | vertical component of velocity (m/s), positive up. |
| bathymetry | (lat, lon) | numeric | depth of seafloor at each grid cell (m, positive up) |

### 10m Wind Velocity
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
| time | (time) | datetime (CF- or ISO-compliant) | time coordinate of field (coordinate variable). Values must be sorted and ascending. |
| U | (lon, lat, time) | numeric | zonal component of velocity (m/s), positive east.
| V | (lon, lat, time) | numeric | meridional component of velocity (m/s), positive north.


### Seawater Density
Encoded as a netcdf dataset, and must adhere to the following specifications:

#### Dimensions
| Name | Data Type | Description |
| --- | --- | --- |
| lon | integer | indexes longitude |
| lat | integer | indexes latitude |
| depth | integer | indexes depth |
| time | integer | indexes time |

#### Variables
| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| lon | (lon) | numeric | longitude of grid cell centers, degrees E, domain [-180, 180) (coordinate variable).  Values must be sorted and equally spaced. |
| lat | (lat) | numeric | latitude of grid cell centers, degrees E, domain [-90, 90] (coordinate variable).  Values must be sorted and equally spaced. |
| depth | (depth) | numeric | depth of grid cell centers, meters, positive up (coordinate variable).  Values must be sorted but can have arbitrary spacing. |
| time | (time) | datetime (CF- or ISO-compliant) | time coordinate of field (coordinate variable). Values must be sorted and ascending. |
| rho | (time, depth, lat, lon) | numeric | density of seawater (kg m^-3)


Notes:
* Non-standard names on any dimension or variable are supported via name mappings passed into `run_advector` in src/run_advector.py.
* Any "packed" variables must have the attributes `scale_factor`, `add_offset`, and `_FillValue` in order to be properly unpacked.  See the [netcdf conventions](https://www.unidata.ucar.edu/software/netcdf/docs/attribute_conventions.html) for more details.
* Domain [0, 360) is supported for longitude but the internal conversion to [-180, 180) is computationally expensive.
