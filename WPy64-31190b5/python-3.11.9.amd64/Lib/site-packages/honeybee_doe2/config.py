"""Global settings and configurations governing the translation process."""

# global parameters that may need to be adjusted or exposed in the future
DOE2_TOLERANCE = 0.03  # current best guess for DOE-2 absolute tolerance in Feet
DOE2_ANGLE_TOL = 1.0  # current best guess for DOE-2 angle tolerance in degrees
FLOOR_LEVEL_TOL = 0.1  # tolerance for grouping Rooms by floor elevations in Feet
GEO_DEC_COUNT = 4  # number of decimal places that all geometry will be rounded
RECT_WIN_SUBD = 0.5  # subdivision distance to rectangularize windows in Feet
DOE2_INTERIOR_BCS = ('Surface', 'Adiabatic', 'OtherSideTemperature')
MIN_LAYER_THICKNESS = 0.003  # the minimum thickness for a material to be valid in meters
GEO_CHARS = 24  # number of original characters used in names of geometry
RES_CHARS = 30  # number of characters used in names of resources (constructions, etc.)
