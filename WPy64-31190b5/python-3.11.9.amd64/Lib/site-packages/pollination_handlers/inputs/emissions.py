"""Handlers to convert different emissions inputs to values."""

from ladybug.location import Location
from honeybee_energy.result.emissions import future_electricity_emissions


def location_to_electricity_emissions(value):
    """Translate a Location object in the US into electricity carbon emissions.

        Args:
            value: Either a ladybug Location object in the USA, which will be used to 
                determine the subregion of the electrical grid or a number
                for the electric grid carbon emissions in kg CO2/MWh.

        Returns:
            float -- electricity emissions in kg CO2/MWh.
    """
    if isinstance(value, Location):
        yr = 2024
        elec_emiss = future_electricity_emissions(value, yr)
        if elec_emiss is None:
            msg = 'Location must be inside the USA in order to be used for carbon ' \
                'emissions estimation.\nPlug in a number for carbon intensity in ' \
                'kg CO2/MWH for locations outside the USA.'
            raise ValueError(msg)
    else:
        try:
            elec_emiss = float(value)
        except TypeError:
            msg = 'Expected location object or number for electricity emissions. ' \
                'Got {}.'.format(type(value))
            raise ValueError(msg)
    return elec_emiss
