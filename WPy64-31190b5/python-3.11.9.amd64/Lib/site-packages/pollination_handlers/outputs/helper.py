import os
import json


def read_sensor_grid_result(
        result_folder, extension, grid_key, is_percent=True, factor=1):
    """Read results from files that align with sensor grids.

    Args:
        result_folder: Path to the folder containing the results.
        extension: Text for the file extension to be read (eg. res).
        grid_key: Text for the key in the grids_info.json that identifies the
            file name of each sensor grid. It is either identifier or full_id.
        is_percent: Boolean to note if the values are intended to be percent, in
            which case a check will be done to ensure no value is greater than
            one hundred.
        factor: An optional number to be multiplied by all of the results.
            This can be used to perform unit conversions or change fractional values
            to percentages. (Default: 1)

    Returns:
        A matrix with each sub-list containing the values for each of the sensor grids.
    """
    # check that the required files are present
    if not os.path.isdir(result_folder):
        raise ValueError('Invalid result folder: %s' % result_folder)
    grid_json = os.path.join(result_folder, 'grids_info.json')
    if not os.path.isfile(grid_json):
        raise ValueError('Result folder contains no grids_info.json.')

    # load the list of grids and gather all of the results
    with open(grid_json) as json_file:
        grid_list = json.load(json_file)
    results = []
    for grid in grid_list:
        grid_id = grid[grid_key]
        sensor_count = grid['count']
        try:
            st_ln = grid['start_ln']
        except KeyError:
            # older version of sensor info
            st_ln = 0

        result_file = os.path.join(result_folder, '{}.{}'.format(grid_id, extension))
        with open(result_file) as inf:
            for _ in range(st_ln):
                next(inf)
            if is_percent:
                results.append(
                    [min(float(next(inf)) * factor, 100) for _ in range(sensor_count)]
                )
            else:
                results.append([float(next(inf)) * factor for _ in range(sensor_count)])

    return results


def read_grid_results(
        result_folder, extension, grid_key, is_percent=True, factor=1):
    """Read results from files that align with sensor grids.
    
    This function should be used if the there is only one value per grid.

    Args:
        result_folder: Path to the folder containing the results.
        extension: Text for the file extension to be read (eg. res).
        grid_key: Text for the key in the grids_info.json that identifies the
            file name of each sensor grid. It is either identifier or full_id.
        is_percent: Boolean to note if the values are intended to be percent, in
            which case a check will be done to ensure no value is greater than
            one hundred.
        factor: An optional number to be multiplied by all of the results.
            This can be used to perform unit conversions or change fractional values
            to percentages. (Default: 1)

    Returns:
        A list containing the value for each of the sensor grids.
    """
    # check that the required files are present
    if not os.path.isdir(result_folder):
        raise ValueError('Invalid result folder: %s' % result_folder)
    grid_json = os.path.join(result_folder, 'grids_info.json')
    if not os.path.isfile(grid_json):
        raise ValueError('Result folder contains no grids_info.json.')

    # load the list of grids and gather all of the results
    with open(grid_json) as json_file:
        grid_list = json.load(json_file)
    results = []
    for grid in grid_list:
        grid_id = grid[grid_key]

        result_file = os.path.join(result_folder, '{}.{}'.format(grid_id, extension))
        with open(result_file) as inf:
            value = inf.readline().rstrip()
            if is_percent:
                results.append(min(float(value) * factor, 100))
            else:
                results.append(float(value) * factor)

    return results
