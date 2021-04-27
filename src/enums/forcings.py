from enum import Enum


class Forcing(Enum):
    """use .name for variable name, .value for human readable name"""
    current = "current"
    wind = "10-meter wind"
    seawater_density = "seawater density"
