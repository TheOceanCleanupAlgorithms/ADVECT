enum ExitCode {
    SUCCESS = 0, NULL_LOCATION = 1, INVALID_LATITUDE = 2, SEAWATER_DENSITY_LOOKUP_FAILURE = 3, INVALID_ADVECTION_SCHEME = -1
};
// positive codes are considered non-fatal, and are reported in outputfiles;
// negative codes are considered fatal, cause host-program termination, and are reserved for internal use.
// if you change these codes, update in src/kernel_wrappers/kernel_constants.py
