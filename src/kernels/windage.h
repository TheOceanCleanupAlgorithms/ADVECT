#ifndef WINDAGE
#define WINDAGE

#include "fields.h"
#include "particle.h"

#define density_air_per_density_water 1.17e-3  // density of air / density of water (van der Mheen 2020 near eq. 8)
#define drag_ratio 1           // drag coefficient in air / drag coefficient in water (van der Mheen 2020 near eq. 8)
#define wind_10cm_per_wind_10m .36156  // wind(10cm)/wind(10m) = (log(.01/.0002) / log(10/.0002)), log wind profile assumption (Charnock 1955),
                                        // taking surface roughness = .0002 (WMO Guide to Instruments and Methods of Observation, 2018 edition, page 211)

vector windage_meters(particle p, field3d wind, double dt, double windage_multiplier);

#endif // WINDAGE
