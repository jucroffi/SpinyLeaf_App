"""Helper functions."""
import json
from pathlib import Path
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from honeybee.model import Model


def model_grid_areas(model, grids_info):
    if isinstance(model, Model):
        hb_model = model
    else:
        hb_model = Model.from_file(model)

    full_ids = [grid_info['full_id'] for grid_info in grids_info]
    sensor_grids = hb_model.properties.radiance.sensor_grids
    grid_areas = []
    for s_grid in sensor_grids:
        if s_grid.identifier in full_ids:
            if s_grid.mesh is not None:
                grid_areas.append(s_grid.mesh.face_areas)
    grid_areas = [np.array(grid) for grid in grid_areas]
    if not grid_areas:
        grid_areas = [None] * len(full_ids)

    return grid_areas


def grid_summary(
        folder: Path, grid_areas: list = None,
        grids_info: list = None, name: str = 'grid_summary',
        grid_metrics: list = None, sub_folder: bool = True
    ):
    """Calculate a grid summary for a single metric.

    Args:
        folder: A folder with results.
        grid_areas: A list of area of each sensor.
        grids_info: Grid information as a dictionary.
        name: Optional filename of grid summary.
        grid_metrics: Additional customized metrics to calculate.
        sub_folder: If set to True it will look for results in all sub-folders
            in the folder input. Else it look for results directly in the folder
            input.
    """
    if sub_folder:
        sub_folders = [sf for sf in folder.iterdir() if sf.is_dir()]
    else:
        sub_folders = [folder]

    # set up the default data types
    dtype_sensor_grid = ('Sensor Grid', 'O')
    dtype_sensor_grid_id = ('Sensor Grid ID', 'O')
    dtype_base = [
        ('Mean', np.float32),
        ('Minimum', np.float32),
        ('Maximum', np.float32),
        ('Uniformity Ratio', np.float32)
    ]
    dtype = []

    # set up default format (for first two columns: str)
    fmt = ['%s', '%s']

    if grids_info is None:
        for sf in sub_folders:
            gi_file = sf.joinpath('grids_info.json')
            if gi_file.exists():
                with open(gi_file) as gi:
                    grids_info = json.load(gi)
                break
        if grids_info is None:
            # if it did not find grids_info.json in any folder
            raise FileNotFoundError(
                f'The file grids_info.json was not found in any folder.')

    if grid_areas is None:
        grid_areas = [None] * len(grids_info)


    dtype.append(dtype_sensor_grid)
    dtype.append(dtype_sensor_grid_id)
    for sf in sub_folders:
        _dtype = []
        _fmt = []
        for dt_b in dtype_base:
            col_name = dt_b[0]
            if sub_folder:
                col_name = '-'.join([sf.stem.upper(), col_name])
            _dtype.append((col_name, np.float32))
            _fmt.append('%.2f')
        dtype.extend(_dtype)
        fmt.extend(_fmt)

        if grid_metrics is not None:
            for grid_metric in grid_metrics:
                if len(grid_metric) == 1:
                    if 'allOf' in grid_metric:
                        _mname = []
                        for gr_m in grid_metric['allOf']:
                            _mname.append(_get_grid_metric_name(gr_m))
                        mname = ' and '.join(_mname)
                    elif 'anyOf' in grid_metric:
                        _mname = []
                        for gr_m in grid_metric['anyOf']:
                            _mname.append(_get_grid_metric_name(gr_m))
                        mname = ' or '.join(_mname)
                    else:
                        mname = _get_grid_metric_name(grid_metric)
                elif len(grid_metric) == 2:
                    _mname = []
                    for k, v in grid_metric.items():
                        _mname.append(_get_grid_metric_name({k: v}))
                    mname = ' and '.join(_mname)
                col_name = mname
                if sub_folder:
                    col_name = '-'.join([sf.stem.upper(), col_name])
                dtype.append((col_name, np.float32))
                fmt.append('%.2f')

    arrays = []
    for grid_info, grid_area in zip(grids_info, grid_areas):
        full_id = grid_info['full_id']
        grid_name = grid_info['name']
        data = [grid_name, full_id]
        for sf in sub_folders:
            grid_files = list(sf.glob(f'{full_id}.*'))
            assert len(grid_files) == 1

            array = np.loadtxt(grid_files[0])
            _mean = array.mean()
            _min = array.min()
            _max = array.max()
            _uniformity_ratio = _min / _mean * 100

            data.extend([_mean, _min, _max, _uniformity_ratio])

            if grid_metrics is not None:
                # get grid metrics
                grid_metrics_data = \
                    _get_grid_metrics(array, grid_metrics, grid_info, grid_area)
                data.extend(grid_metrics_data)

        arrays.append(tuple(data))

    # create structured array
    if is_gpu:
        struct_array = None
    else:
        struct_array = np.array(arrays, dtype=dtype)

    header = [dt[0] for dt in dtype]
    # write header to file
    with open(folder.joinpath(f'{name}.csv'), 'w') as grid_summary_file:
        grid_summary_file.write(','.join(header))
    # write structured array to grid_summary_file
    with open(folder.joinpath(f'{name}.csv'), 'a') as grid_summary_file:
        grid_summary_file.write('\n')
        if is_gpu:
            # CuPy doesn't support structured arrays; manually format rows
            for row in arrays:
                row_str = ','.join(fmt_val % val for fmt_val, val in zip(fmt, row))
                grid_summary_file.write(row_str + '\n')
        else:
            np.savetxt(grid_summary_file, struct_array, delimiter=',', fmt=fmt)

    return grid_summary_file


def _calculate_percentage(gr_metric_bool, grid_info, grid_area=None):
    """Calculate percentage of floor area where True.

    Args:
        gr_metric_bool: A NumPy array of booleans.
        grid_info: Grid information.
        grid_area: A NumPy array of area for each sensor. (Default: None).
    
    Returns:
        The percentage of floor area where gr_metric_bool is True.
    """
    if grid_area is not None:
        gr_metric_pct = \
            grid_area[gr_metric_bool].sum() / grid_area.sum() * 100
    else:
        gr_metric_pct = \
            gr_metric_bool.sum() / grid_info['count'] * 100
    return gr_metric_pct


def _logical_operator(keyword):
    lg = {
        'minimum': '>',
        'exclusiveMinimum': '>=',
        'maximum': '<',
        'exclusiveMaximum': '<='
    }
    return lg[keyword]


def _get_grid_metric_name(grid_metric):
    if 'minimum' in grid_metric:
        return f'{_logical_operator("minimum")}{grid_metric["minimum"]}'
    elif 'exclusiveMinimum' in grid_metric:
        return f'{_logical_operator("exclusiveMinimum")}{grid_metric["exclusiveMinimum"]}'
    elif 'maximum' in grid_metric:
        return f'{_logical_operator("maximum")}{grid_metric["maximum"]}'
    elif 'exclusiveMaximum' in grid_metric:
        return f'{_logical_operator("exclusiveMaximum")}{grid_metric["exclusiveMaximum"]}'


def _numeric_type(array, gr_metric):
    if 'minimum' in gr_metric:
        gr_metric_bool = array > gr_metric['minimum']
    elif 'exclusiveMinimum' in gr_metric:
        gr_metric_bool = array >= gr_metric['minimum']
    elif 'maximum' in gr_metric:
        gr_metric_bool = array < gr_metric['maximum']
    elif 'exclusiveMaximum' in gr_metric:
        gr_metric_bool = array <= gr_metric['exclusiveMaximum']
    return gr_metric_bool


def _grid_summary_all_any(array, gr_metric, grid_info, grid_area, keyword):
    gr_metric_arrays = []
    for gr_m in gr_metric[keyword]:
        assert len(gr_m) == 1
        gr_metric_arrays.append(_numeric_type(array, gr_m))
    if keyword == 'allOf':
        gr_metric_bool = np.all(gr_metric_arrays, axis=0)
    else:
        gr_metric_bool = np.any(gr_metric_arrays, axis=0)
    gr_metric_pct = \
        _calculate_percentage(gr_metric_bool, grid_info, grid_area)
    return gr_metric_pct


def _get_grid_metrics(array, grid_metrics, grid_info, grid_area):
    grid_metrics_data = []
    for gr_metric in grid_metrics:
        if len(gr_metric) == 1:
            if 'allOf' in gr_metric:
                gr_metric_pct = \
                    _grid_summary_all_any(
                        array, gr_metric, grid_info, grid_area, 'allOf')
            elif 'anyOf' in gr_metric:
                gr_metric_pct = \
                    _grid_summary_all_any(
                        array, gr_metric, grid_info, grid_area, 'anyOf')
            else:
                gr_metric_bool = _numeric_type(array, gr_metric)
                gr_metric_pct = \
                    _calculate_percentage(gr_metric_bool, grid_info, grid_area)
        elif len(gr_metric) == 2:
            gr_metric_arrays = []
            for k, threshold in gr_metric.items():
                gr_metric_arrays.append(_numeric_type(array, {k: threshold}))
            gr_metric_bool = np.all(gr_metric_arrays, axis=0)
            gr_metric_pct = \
                _calculate_percentage(gr_metric_bool, grid_info, grid_area)
        grid_metrics_data.append(gr_metric_pct)
    return grid_metrics_data
