"""Utility functions for sensor grids folder."""
import os
import json


def redistribute_sensors(
    input_folder, output_folder, grid_count, min_sensor_count=2000,
    extension='pts', grid_info=None, verbose=False
):
    """Create new sensor grids folder with evenly distributed sensors.

    This function creates a new folder with evenly distributed sensor grids. The folder
    will include a ``_redist_info.json`` file which has the information to recreate the
    original input files from this folder and the results generated based on the grids
    in this folder.

    ``_redist_info.json`` file includes an array of JSON objects. Each object has the
    ``id`` or the original file and the distribution information. The distribution
    information includes the id of the new files that the sensors has been distributed
    to and the start and end line in the target file.

    This file is being used to restructure the data that is generated based on the newly
    created sensor grids.

    .. code-block:: python

        [
          {
            "id": "room_1",
            "dist_info": [
              {"id": 0, "st_ln": 0, "end_ln": 175},
              {"id": 1, "st_ln": 0, "end_ln": 21}
            ]
          },
          {
            "id": "room_2",
            "dist_info": [
              {"id": 1, "st_ln": 22, "end_ln": 135}
            ]
          }
        ]

    Args:
        input_folder: Input sensor grids folder.
        output_folder: A new folder to write the newly created files.
        grid_count: Number of output sensor grids to be created. This number
            is usually equivalent to the number of processes that will be used to run
            the simulations in parallel.
        min_sensor_count: Minimum number of sensors in each output grid. Use this number
            to ensure the number of sensors in output grids never gets very small. To
            ignore this limitation set the value to 1. This value always takes precedence
            over grid_count. Default: 2000.
        extension: Extension of the files to collect data from. Default is ``pts`` for
            sensor files. Another common extension is ``csv`` for generic data sets.
        grid_info: Optional list of dictionaries with grid information. Use this instead
            of the expected _info.json file in the input_folder.
        verbose: Set to True to get verbose reporting. Default: False.

    Returns:
        A tuple with three elements

        - grid_count: Number of output sensor grids. Keep in mind that this number can
          be adjusted based on the min_sensor_count.

        - sensor_per_grid: Number of sensors in each newly created sensor grid.

        - out_grid_info: Grid information as written to _info.json.
    """
    if grid_info is None:
        info_file = os.path.join(input_folder, '_info.json')
        assert os.path.isfile(info_file), \
            'Failed to find the _info.json file. This file should be located inside the ' \
            'input folder.'
        with open(info_file) as inf:
            data = json.load(inf)
    else:
        #TODO: add check that validates grid_info
        data = grid_info
    total_count = sum(grid['count'] for grid in data)
    sensor_per_grid = int(round(total_count / grid_count)) or 1
    if sensor_per_grid < min_sensor_count:
        # re-calculate based on minimum sensor counts
        grid_count = int(round(total_count / min_sensor_count)) or 1
        sensor_per_grid = int(round(total_count / grid_count))

    # collect data from original folder
    input_grid_files = []
    dist_info = []
    for d in data:
        input_grid_files.append(os.path.join(input_folder, d['full_id']))
        dist_info.append({'identifier': d['full_id'], 'dist_info': []})
    input_grids_iter = iter(input_grid_files)

    def get_next_input_grid():
        id_ = next(input_grids_iter) + '.' + extension
        if verbose:
            print('Started reading from %s' % id_)
        return open(id_)

    def get_target_file(index):
        outf = os.path.join(output_folder, '{}.{}'.format(index, extension))
        if verbose:
            print('Started writing to %s' % outf)
        return open(outf, 'w')

    # create output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    extra_sensors = False
    out_grid_info = []
    outf_index = 0
    outf = get_target_file(outf_index)
    line_out_count = 0
    for i in range(len(input_grid_files)):
        # get an input sensor grid
        inf = get_next_input_grid()
        # set the input grid count and line count for input grid
        input_grid_count = data[i]['count']
        line_grid_count = 0
        # set the information for output file index and start line
        dist_info[i]['dist_info'].append(
            {'identifier': outf_index, 'st_ln': line_out_count}
        )
        # start writing the lines to the target file until the number of lines is
        # equal to the number of sensors that should be written to each grid file
        for line in inf:
            outf.write(line)
            line_out_count += 1
            line_grid_count += 1
            if line_out_count == sensor_per_grid:
                # the current file is full. add end line for input file
                dist_info[i]['dist_info'][-1]['end_ln'] = sensor_per_grid - 1
                # move on to next out file
                out_data = {
                    'name': str(outf_index),
                    'identifier': str(outf_index),
                    'full_id': str(outf_index),
                    'group': '',
                    'count': line_out_count
                }
                outf_index += 1
                if outf_index == grid_count:
                    # This was the last file and more sensors are left.
                    # Add the remainder of the sensors to this file and close the file.
                    extra_sensors = True
                    counter = 0
                    for line in inf:
                        counter += 1
                        outf.write(line)
                    inf.close()
                    # adjust the end line in dist_info
                    dist_info[i]['dist_info'][-1]['end_ln'] += counter
                    out_data['count'] += counter
                    out_grid_info.append(out_data)
                    line_out_count += counter
                    outf_index -= 1
                    break
                line_out_count = 0
                outf.close()
                out_grid_info.append(out_data)
                # open a new file
                outf = get_target_file(outf_index)
                if not line_grid_count == input_grid_count:
                    dist_info[i]['dist_info'].append(
                        {'identifier': outf_index, 'st_ln': line_out_count}
                    )
            # special case if outf_index == grid_count before all input grids have been
            # redistributed
            if extra_sensors and line_grid_count == input_grid_count:
                # adjust the counter in the last grid
                out_grid_info[-1]['count'] += line_grid_count
        if 'end_ln' not in dist_info[i]['dist_info'][-1]:
            dist_info[i]['dist_info'][-1]['end_ln'] = line_out_count - 1
        inf.close()
    if not extra_sensors:
        # add the information for the last sensor grid
        out_data = {
            'name': str(outf_index),
            'identifier': str(outf_index),
            'full_id': str(outf_index),
            'group': '',
            'count': line_out_count
        }
        out_grid_info.append(out_data)

    dist_info_file = os.path.join(output_folder, '_redist_info.json')
    with open(dist_info_file, 'w') as dist_out_file:
        json.dump(dist_info, dist_out_file, indent=2)

    info_file = os.path.join(output_folder, '_info.json')
    with open(info_file, 'w') as dist_out_file:
        json.dump(out_grid_info, dist_out_file, indent=2)

    print(
        'Distributed %d sensors among %d grids with %d sensors each.' % (
            total_count, grid_count, sensor_per_grid
            )
    )

    return grid_count, sensor_per_grid, out_grid_info


def restore_original_distribution(
        input_folder, output_folder, extension='pts', dist_info=None):
    """Restructure files to the original distribution based on the distribution info.

    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder
        extension: Extension of the files to collect data from. Default is ``pts`` for
            sensor files. Another common extension is ``ill`` for the results of daylight
            studies.
        dist_info: Path to dist_info.json file. If None, the function will try to load
            ``_redist_info.json`` file from inside the input_folder. (Default: None).

    """
    if not dist_info:
        _redist_info_file = os.path.join(input_folder, '_redist_info.json')
    else:
        _redist_info_file = dist_info

    assert os.path.isfile(_redist_info_file), 'Failed to find %s' % _redist_info_file

    with open(_redist_info_file) as inf:
        data = json.load(inf)

    # create output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for f in data:
        out_file = os.path.join(output_folder, '%s.%s' % (f['identifier'], extension))
        # ensure the new folder is created. in case the identifier has a subfolder
        parent_folder = os.path.dirname(out_file)
        if not os.path.isdir(parent_folder):
            os.mkdir(parent_folder)
        with open(out_file, 'w') as outf:
            for src_info in f['dist_info']:
                src_file = os.path.join(
                    input_folder, '%s.%s' % (src_info['identifier'], extension)
                )
                st = src_info['st_ln']
                end = src_info['end_ln']
                with open(src_file) as srf:
                    # remove the header if it is there
                    first_line = next(srf)
                    if first_line[:10] == '#?RADIANCE':
                        for line in srf:
                            if line[:7] == 'FORMAT=':
                                # pass next empty line
                                next(srf)
                                first_line = next(srf)
                                break
                            continue
                    # go to the start line and write it
                    for _ in range(st):
                        first_line = next(srf)
                    outf.write(first_line)
                    # write the other lines to the output file
                    for _ in range(end - st):
                        outf.write(next(srf))


def parse_grid_info(info_file):
    """Parse sensor grid information from a info json file. This information
    typically contains full_id, name, identifier, group and count. In the case
    of _model_grids_info json, it will also contain the starting line (start_ln)

    Args:
        info_file: Path to the grid info file.

    Returns:
        A list containing the information about sensor grids.
    """
    if not os.path.isfile(info_file):
        return []

    with open(info_file) as inf:
        return json.load(inf)


def parse_grid_json(grid_json_file):
    """
    Parse a grid json file and return the corresponding dictionary
    Args:
        grid_json_file: Input path of the grid json file.

    Returns:
        If path exists, return the grid info dictionary. Else return None.
    """
    if not os.path.isfile(grid_json_file):
        return None

    with open(grid_json_file) as grid_data:
        return json.load(grid_data)
