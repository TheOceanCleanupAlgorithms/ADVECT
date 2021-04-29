# Forcing Data Specifications

##2D Kernel
The 2D Kernel can accept ocean current velocity and 10-meter wind velocity.
### Surface Ocean Current Velocity
Encoded as a netcdf dataset, and must adhere to the following specifications.  Note: if a depth dimension is present, the level closest to depth=0 is extracted.

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
| U | (time, lat, lon) | numeric | zonal component of velocity (m/s), positive east.
| V | (time, lat, lon) | numeric | meridional component of velocity (m/s), positive north. |

### 10m Wind Velocity
Encoded as a netcdf dataset, and must adhere to the following specifications.  Note: if a depth dimension is present, the level closest to depth=0 is extracted.

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

##3D Kernel
The 3D Kernel can accept ocean current velocity, 10-meter wind velocity, and seawater density.

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

### 10m Wind Velocity
Exact same specifications as outlined above for the 2D Kernel.  Note that the near-surface, turbulent wind-mixing algorithm assumes that each wind data is an average over a timeframe of ~1 day, as it assumes the sea state is fully developed based on the closest wind observation.  If you pass in wind data with sub-daily intervals along the time dimension, the wind mixing will become less realistic.

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


##Notes:
* Non-standard names on any dimension or variable are supported via name mappings passed into the top level APIs, `run_advector_2d.py` and `run_avector_3d.py`.
* Any "packed" variables must have the attributes `scale_factor`, `add_offset`, and `_FillValue` in order to be properly unpacked.  See the [netcdf conventions](https://www.unidata.ucar.edu/software/netcdf/docs/attribute_conventions.html) for more details.
* If longitude domain is detected as [0, 360), it will be internally converted to [-180, 180).  However, this is computationally expensive and will slow down data loading.
* If detected positive-down, any depth dimensions will be automatically converted to positive-up.
* Any depth dimensions will be automatically sorted ascending.
