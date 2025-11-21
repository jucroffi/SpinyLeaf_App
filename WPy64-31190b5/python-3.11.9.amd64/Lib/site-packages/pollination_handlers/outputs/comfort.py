"""Handlers for comfort mapping simulation."""
from .helper import read_sensor_grid_result


def read_comfort_percent_from_folder(result_folder):
    """Read comfort percent values from a folder with .csv result files.

    The result with be a matrix with each sub-list containing the percent
    values for each of the sensor grids.
    """
    try:
        return read_sensor_grid_result(result_folder, 'csv', 'full_id')
    except KeyError:  # older specification
        return read_sensor_grid_result(result_folder, 'csv', 'id')
