"""Handlers to convert inputs that accept ladybug data collections."""
import os
import json

from ladybug.datacollection import BaseCollection

from .helper import get_tempfile, write_values_to_csv


def value_or_data_to_str(value):
    """Translate a single numerical value or data collection into a string.

    Args:
        value: Either a single numerical value or a data collection.

    Returns:
        str -- string version of the number or JSON array string of data
            collection values.
    """
    if isinstance(value, str):
        if value != 'None':
            try:  # first check to see if it's a valid number
                float(value)
            except ValueError:  # maybe it's a JSON array of numbers
                loaded_data = json.loads(value)
                assert isinstance(loaded_data, list), \
                    'Data string must be either a number or an array.'
    elif isinstance(value, (float, int)):
        value = str(value)
    elif isinstance(value, BaseCollection):
        start_values = ['{0:.3f}'.format(v) for v in value.values]
        final_values = []
        for v in start_values:
            new_v = v.rstrip('0') 
            new_v = '{}0'.format(new_v) if new_v.endswith('.') else new_v
            final_values.append(new_v)
        value = str(final_values).replace('\'', '')
    else:
        raise ValueError(
            'Excpected a single number or a data collection. Not {}.'.format(type(value))
        )
    return value


def value_or_data_to_file(value, file_name=None):
    """Translate a single numerical value or data collection into a file.

    Args:
        value: Either a single numerical value or a data collection.
        file_name: Name for the file. If None, a unique one will be generated.

    Returns:
        file -- a file containing the input value with one row per value.
    """
    values = None
    if value is None:
        pass
    elif isinstance(value, str):
        if not os.path.isfile(value):
            try:  # first check to see if it's a valid number
                values = [float(value)]
            except ValueError:  # maybe it's a JSON array of numbers
                values = json.loads(value)
                assert isinstance(values, list), \
                    'Data string must be either a number or an array.'
    elif isinstance(value, (float, int)):
        values = [value]
    elif isinstance(value, BaseCollection):
        values = value.values
    else:
        raise ValueError(
            'Excpected a single number, a data collection, or a file. '
            'Not {}.'.format(type(value))
        )
    if values is not None:
        str_values = [str(v) for v in values]
        return write_values_to_csv(get_tempfile('csv', file_name), str_values)
    return value


def value_or_data_to_air_speed_file(value):
    """Translate a single numerical value or data collection into an air_speed.csv file.
    """
    return value_or_data_to_file(value, 'air_speed')


def value_or_data_to_met_file(value):
    """Translate a single numerical value or data collection into a met.csv file."""
    return value_or_data_to_file(value, 'met')


def value_or_data_to_clo_file(value):
    """Translate a single numerical value or data collection into a clo.csv file."""
    return value_or_data_to_file(value, 'clo')
