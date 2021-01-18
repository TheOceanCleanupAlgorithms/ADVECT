#ifndef PHYSICAL_CONSTANTS
#define PHYSICAL_CONSTANTS

#define ACC_GRAVITY -9.81  // m s^-2
#define DENSITY_SEAWATER  1025  // kg m^-3, typical value for surface density (no citation, this is well known)
#define KINEMATIC_VISCOSITY_SEAWATER 0.00000118  // m^2 s^-1, typical value (ITTC fresh water and seawater properties,
                                            // https://ittc.info/media/4048/75-02-01-03.pdf)
#define DENSITY_SURFACE_AIR 1.225  // kg m^-3, according to International Standard Atmosphere (ISO 2533:1975)
#define VON_KARMAN_CONSTANT .4
#define MAX_RECORDED_SIGNIFICANT_WAVE_HEIGHT 20 // m, approximate world record as of 2020 (https://wmo.asu.edu/content/World-Highest-Wave-Buoy)

#endif // PHYSICAL_CONSTANTS
