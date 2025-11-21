"""honeybee radiance daylight postprocessing commands."""
from pathlib import Path
import sys
import os
import logging
import json
import click
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ladybug.location import Location
from ladybug.wea import Wea
from honeybee_radiance_postprocess.results.annual_daylight import AnnualDaylight
from honeybee_radiance_postprocess.results.annual_irradiance import AnnualIrradiance
from honeybee_radiance_postprocess.metrics import da_array2d, cda_array2d, \
    udi_array2d, udi_lower_array2d, udi_upper_array2d
from honeybee_radiance_postprocess.reader import binary_to_array

from ..annual import occupancy_schedule_8_to_6
from ..dynamic import DynamicSchedule
from ..en17037 import en17037_to_folder
from ..util import filter_array
from .two_phase import two_phase
from .leed import leed
from .abnt import abnt
from .well import well
from .breeam import breeam
from ..helper import model_grid_areas, grid_summary

_logger = logging.getLogger(__name__)


@click.group(help='Commands to post-process Radiance results.')
def post_process():
    pass

post_process.add_command(two_phase)
post_process.add_command(leed)
post_process.add_command(abnt)
post_process.add_command(well)
post_process.add_command(breeam)


@post_process.command('annual-daylight')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--threshold', '-t', help='Threshold illuminance level for daylight autonomy.',
    default=300, type=int, show_default=True
)
@click.option(
    '--lower-threshold', '-lt',
    help='Minimum threshold for useful daylight illuminance.', default=100, type=int,
    show_default=True
)
@click.option(
    '--upper-threshold', '-ut',
    help='Maximum threshold for useful daylight illuminance.', default=3000, type=int,
    show_default=True
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def annual_metrics(
    folder, schedule, threshold, lower_threshold, upper_threshold, states,
    grids_filter, sub_folder
):
    """Compute annual metrics in a folder and write them in a subfolder.

    \b
    This command generates 5 files for each input grid.
        da/{grid-name}.da -> Daylight Autonomy
        cda/{grid-name}.cda -> Continuos Daylight Autonomy
        udi/{grid-name}.udi -> Useful Daylight Illuminance
        udi_lower/{grid-name}_upper.udi -> Upper Useful Daylight Illuminance
        udi_upper/{grid-name}_lower.udi -> Lower Useful Daylight Illuminance

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        results = AnnualDaylight(folder, schedule=schedule)
        results.annual_metrics_to_folder(
            sub_folder, threshold=threshold, min_t=lower_threshold,
            max_t=upper_threshold, states=states, grids_filter=grids_filter
        )
    except Exception:
        _logger.exception('Failed to calculate annual metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-daylight-en17037')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.argument(
    'schedule',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub_folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='en17037'
)
def annual_en17037_metrics(
    folder, schedule, states, grids_filter, sub_folder
):
    """Compute annual EN 17037 metrics in a folder and write them in a subfolder.

    \b
    This command generates multiple files for each input grid. Files for target
    illuminance and minimum illuminance will be calculated for three levels of
    recommendation: minimum, medium, high.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
        schedule: Path to an annual schedule file. Values should be 0-1
            separated by new line. This should be a daylight hours schedule.
    """
    with open(schedule) as hourly_schedule:
        schedule = [int(float(v)) for v in hourly_schedule]

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        en17037_to_folder(
            folder, schedule, states=states, grids_filter=grids_filter,
            sub_folder=sub_folder)
    except Exception:
        _logger.exception('Failed to calculate annual EN 17037 metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('average-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def average_values(
    folder, hoys_file, states, grids_filter, total, sub_folder
):
    """Get average values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = AnnualDaylight(folder)
        results.average_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate average values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('median-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def median_values(
    folder, hoys_file, states, grids_filter, total, sub_folder
):
    """Get median values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = AnnualDaylight(folder)
        results.median_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate median values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('cumulative-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def cumulative_values(
    folder, hoys_file, states, grids_filter, total, sub_folder
):
    """Get cumulative values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = AnnualDaylight(folder)
        results.cumulative_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate cumulative values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('peak-values')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--hoys-file', '-h', help='Path to an HOYs file. Values must be separated by '
    'new line. If not provided the data will not be filtered by HOYs.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--coincident/--non-coincident', is_flag=True, default=False, show_default=True,
    help='Boolean to indicate whether output values represent the the peak value for '
    'each sensor throughout the entire analysis (False) or they represent the highest '
    'overall value across each sensor grid at a particular timestep (True).'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def peak_values(
    folder, hoys_file, states, grids_filter, total, coincident, sub_folder
):
    """Get peak values for each sensor over a given period.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if hoys_file:
            with open(hoys_file) as hoys:
                hoys = [float(h) for h in hoys.readlines()]
        else:
            hoys = []

        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = AnnualDaylight(folder)
        results.peak_values_to_folder(
            sub_folder, hoys=hoys, states=states, grids_filter=grids_filter,
            coincident=coincident, res_type=res_type)
    except Exception:
        _logger.exception('Failed to calculate peak values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-to-data')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sensor-index', '-si', help='A JSON file with a dictionary of sensor indices '
    'for each grid. If not provided all sensors will be used.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--output-file', '-f', help='Optional file to output the JSON strings of '
    'the data collections. By default, it will be printed to stdout',
    type=click.File('w'), default='-', show_default=True
)
def annual_to_data(
    folder, states, grids_filter, sensor_index, total, output_file
):
    """Get annual data collections as JSON files.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    if states:
        states = DynamicSchedule.from_json(states)

    if sensor_index:
        with open(sensor_index) as json_file:
            sensor_index = json.load(json_file)

    res_type = 'total' if total is True else 'direct'

    try:
        results = AnnualDaylight(folder)
        data_cs, grids_info, sensor_index = results.annual_data(
            states=states, grids_filter=grids_filter,
            sensor_index=sensor_index, res_type=res_type)
        data_colls = [[data.to_dict() for data in data_list] for data_list in data_cs]
        output_file.write(json.dumps(data_colls))
    except Exception:
        _logger.exception('Failed to create data collections.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('point-in-time')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.argument(
    'hoy', type=click.FLOAT
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states '
    'are not provided the default states will be used for any aperture groups.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--total/--direct', is_flag=True, default=True, help='Switch between total '
    'and direct results. Default is total.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def point_in_time(
    folder, hoy, states, grids_filter, total, sub_folder
):
    """Get point in time values.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
        hoy: An HOY (point-in-time) for which to get the point-in-time values.
    """
    try:
        if states:
            states = DynamicSchedule.from_json(states)

        res_type = 'total' if total is True else 'direct'

        results = AnnualDaylight(folder)
        results.point_in_time_to_folder(
            sub_folder, datetime=hoy, states=states, grids_filter=grids_filter,
            res_type=res_type)
    except Exception:
        _logger.exception('Failed to point in time values.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-sunlight-exposure')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--direct-threshold', '-dt', help='The threshold that determines if a '
    'sensor is overlit.',
    default=1000, type=float, show_default=True
)
@click.option(
    '--occ-hours', '-oh', help='The number of occupied hours that cannot '
    'receive more than the direct_threshold.', default=250, type=int,
    show_default=True
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def annual_sunlight_exposure(
    folder, schedule, direct_threshold, occ_hours, states, grids_filter,
    sub_folder
):
    """Compute annual sunlight exposure in a folder and write them in a subfolder.

    \b
    This command generates 2 files for each input grid.
        ase/{grid-name}.ase -> Annual Sunlight Exposure
        hours_above/{grid-name}.hours -> Number of overlit hours for each sensor

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        results = AnnualDaylight(folder, schedule=schedule)
        results.annual_sunlight_exposure_to_folder(
            sub_folder, direct_threshold=direct_threshold, occ_hours=occ_hours,
            states=states, grids_filter=grids_filter
        )
    except Exception:
        _logger.exception('Failed to calculate annual sunlight exposure.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-daylight-file')
@click.argument(
    'file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'sun-up-hours',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--threshold', '-t', help='Threshold illuminance level for daylight autonomy.',
    default=300, type=int, show_default=True
)
@click.option(
    '--lower-threshold', '-lt',
    help='Minimum threshold for useful daylight illuminance.', default=100, type=int,
    show_default=True
)
@click.option(
    '--upper-threshold', '-ut',
    help='Maximum threshold for useful daylight illuminance.', default=3000, type=int,
    show_default=True
)
@click.option(
    '--study-info',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help='Optional study info file. This option is needed if the time step is '
    'larger than 1.'
)
@click.option(
    '--grid-name', '-gn', help='Optional name of each metric file.',
    default=None, show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def annual_metrics_file(
    file, sun_up_hours, schedule, threshold, lower_threshold, upper_threshold,
    study_info, grid_name, sub_folder
):
    """Compute annual metrics for a single file and write the metrics in a
    subfolder.

    \b
    This command generates 5 files for each input grid.
        da/{grid-name}.da -> Daylight Autonomy
        cda/{grid-name}.cda -> Continuos Daylight Autonomy
        udi/{grid-name}.udi -> Useful Daylight Illuminance
        udi_lower/{grid-name}_upper.udi -> Upper Useful Daylight Illuminance
        udi_upper/{grid-name}_lower.udi -> Lower Useful Daylight Illuminance

    \b
    Args:
        file: Annual illuminance file. This can be either a NumPy file or a
            binary Radiance file.
        sun_up_hours: A file with the sun up hours of the study.
    """
    file = Path(file)
    # load file to array
    try:
        array = np.load(file)
    except Exception:
        array = binary_to_array(file)

    if study_info and os.path.isfile(study_info):
        with open(study_info) as si_file:
            study_info = json.load(si_file)
        timestep = study_info['timestep']
        study_hours = study_info['study_hours']
    else:
        timestep = 1
        study_hours = \
            Wea.from_annual_values(Location(), [0] * 8760, [0] * 8760).hoys

    # read sun up hours
    sun_up_hours = np.loadtxt(sun_up_hours)
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = occupancy_schedule_8_to_6(timestep=timestep)

    if grid_name is None:
        grid_name = file.stem

    sun_up_hours_mask =  np.where(np.isin(np.array(study_hours), np.array(sun_up_hours)))[0]
    sun_down_hours_mask =  np.where(~np.isin(np.array(study_hours), np.array(sun_up_hours)))[0]
    occ_mask = np.array(schedule, dtype=int)[sun_up_hours_mask]
    sun_down_occ_hours =  np.array(schedule, dtype=int)[sun_down_hours_mask].sum()
    total_hours = sum(schedule)

    array_filter = np.apply_along_axis(
        filter_array, 1, array, mask=occ_mask)

    try:
        da = da_array2d(array_filter, total_occ=total_hours, threshold=threshold)
        cda = cda_array2d(array_filter, total_occ=total_hours, threshold=threshold)
        udi = udi_array2d(
            array_filter, total_occ=total_hours, min_t=lower_threshold,
            max_t=upper_threshold)
        udi_lower = udi_lower_array2d(
            array_filter, total_occ=total_hours, min_t=lower_threshold,
            sun_down_occ_hours=sun_down_occ_hours)
        udi_upper = udi_upper_array2d(
            array_filter, total_occ=total_hours, max_t=upper_threshold)

        sub_folder = Path(sub_folder)
        pattern = {
            'da': da, 'cda': cda, 'udi_lower': udi_lower, 'udi': udi,
            'udi_upper': udi_upper
        }
        for metric, data in pattern.items():
            metric_folder = sub_folder.joinpath(metric)
            extension = metric.split('_')[0]
            output_file = metric_folder.joinpath(f'{grid_name}.{extension}')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, data, fmt='%.2f')
    except Exception:
        _logger.exception('Failed to calculate annual metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('grid-summary')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--model', '-m', help='An optional HBJSON model file. This will be used to '
    'find the area of the grids. The area is used when calculating percentages '
    'of floor area.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-info', '-gi', help='An optional JSON file with grid information. '
    'If no file is provided the command will look for a file in the folder.',
    default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--name', '-n', help='Optional filename of grid summary.',
    type=str, default='grid_summary', show_default=True
)
@click.option(
    '--grid-metrics', '-gm', help='An optional JSON file with additional '
    'custom metrics to calculate.', default=None, show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--sub-folder/--main-folder', is_flag=True, default=True,
    help='If sub-folder is selected it will look for any sub-folders in the '
    'folder argument. If main-folder is selected it will look for result files '
    'in the folder argument.'
)
def grid_summary_metric(
    folder, model, grids_info, name, grid_metrics, sub_folder
):
    """Calculate a grid summary.

    If the grids info file is omitted it is a requirement that there is a grids
    info file in the main folder or in each sub folder.

    \b
    Args:
        folder: A folder with results.
    """
    try:
        # create Path object
        folder = Path(folder)

        # get grids information
        if grids_info and Path(grids_info).is_file():
            with open(grids_info) as gi:
                grids_info = json.load(gi)
        else:
            grids_info = None

        # get grid metrics
        if grid_metrics and Path(grid_metrics).is_file():
            with open(grid_metrics) as gm:
                grid_metrics = json.load(gm)
        else:
            grid_metrics = None

        # check to see if there is a HBJSON with sensor grid meshes for areas
        if grids_info and model:
            grid_areas = model_grid_areas(model, grids_info)
        else:
            grid_areas = None

        grid_summary(
            folder, grid_areas=grid_areas, grids_info=grids_info, name=name,
            grid_metrics=grid_metrics, sub_folder=sub_folder)

    except Exception:
        _logger.exception('Failed to calculate grid summary.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-uniformity-ratio')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--threshold', '-t', help='A threshold for the uniformity ratio. Defaults '
    'to 0.5.',
    default=0.5, type=click.FloatRange(0, 1), show_default=True
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write '
    'annual uniformity ratio.', default='annual_uniformity_ratio'
)
def annual_uniformity_ratio(
    folder, schedule, threshold, states, grids_filter, sub_folder
):
    """Calculate annual uniformity ratio and write it to a folder.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        results = AnnualDaylight(folder, schedule=schedule)
        results.annual_uniformity_ratio_to_folder(
            sub_folder, threshold=threshold, states=states,
            grids_filter=grids_filter
        )
    except Exception:
        _logger.exception('Failed to calculate annual uniformity ratio.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('annual-irradiance')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--schedule', '-sch', help='Path to an annual schedule file. Values should be 0-1 '
    'separated by new line. If not provided an 8-5 annual schedule will be created.',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--states', '-st', help='A JSON file with a dictionary of states. If states are not '
    'provided the default states will be used for any aperture groups.', default=None,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to write output '
    'metric files.', default='metrics'
)
def annual_irradiance_metrics(
    folder, schedule, states, grids_filter, sub_folder
):
    """Compute irradiance metrics in a folder and write them in a subfolder.

    \b
    This command generates 3 files for each input grid.
        average_irradiance/{grid-name}.res -> Average Irradiance (W/m2)
        peak_irradiance/{grid-name}.res -> Peak Irradiance (W/m2)
        cumulative_radiation/{grid-name}.res -> Cumulative Radiation (kWh/m2)

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            irradiance recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
    """
    # optional input - only check if the file exist otherwise ignore
    if schedule and os.path.isfile(schedule):
        with open(schedule) as hourly_schedule:
            schedule = [int(float(v)) for v in hourly_schedule]
    else:
        schedule = None

    if states:
        states = DynamicSchedule.from_json(states)

    try:
        results = AnnualIrradiance(folder, schedule=schedule)
        results.annual_metrics_to_folder(
            sub_folder, states=states, grids_filter=grids_filter
        )
    except Exception:
        _logger.exception('Failed to calculate annual irradiance metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('convert-to-binary')
@click.argument(
    'input-matrix', type=click.Path(exists=True, file_okay=True, resolve_path=True)
)
@click.option(
    '--minimum', type=float, default='-inf', help='Minimum range for values to be '
    'converted to 1.'
)
@click.option(
    '--maximum', type=float, default='+inf', help='Maximum range for values to be '
    'converted to 1.'
)
@click.option(
    '--include-max/--exclude-max', is_flag=True, help='A flag to include the maximum '
    'threshold itself. By default the threshold value will be included.', default=True
)
@click.option(
    '--include-min/--exclude-min', is_flag=True, help='A flag to include the minimum '
    'threshold itself. By default the threshold value will be included.', default=True
)
@click.option(
    '--comply/--reverse', is_flag=True, help='A flag to reverse the selection logic. '
    'This is useful for cases that you want to all the values outside a certain range '
    'to be converted to 1. By default the input logic will be used as is.', default=True
)
@click.option(
    '--name', '-n', help='Name of output file.', default='binary',
    show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def convert_matrix_to_binary(
    input_matrix, minimum, maximum, include_max, include_min, comply, name, output_folder
):
    """Postprocess a Radiance matrix and convert it to 0-1 values.

    \b
    This command is useful for translating Radiance results to outputs like
    sunlight hours. Input matrix must be in ASCII or binary format. The input
    Radiance file must have a header.

    Args:
        input-matrix: A Radiance matrix file.
    """
    array = binary_to_array(input_matrix)
    minimum = float(minimum)
    maximum = float(maximum)
    try:
        if include_max and include_min:
            boolean_array = (array >= minimum) & (array <= maximum)
        elif not include_max and not include_min:
            boolean_array = (array > minimum) & (array < maximum)
        elif include_max and not include_min:
            boolean_array = (array > minimum) & (array <= maximum)
        elif not include_max and include_min:
            boolean_array = (array >= minimum) & (array < maximum)

        if not comply:
            # this will invert the boolean array
            boolean_array = ~boolean_array
        
        binary_array = boolean_array.astype(int)
        output_file = Path(output_folder, name)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, binary_array)
    except Exception:
        _logger.exception('Failed to convert the input file to binary format.')
        sys.exit(1)
    else:
        sys.exit(0)


@post_process.command('direct-sun-hours')
@click.argument(
    'input-matrix', type=click.Path(exists=True, file_okay=True, resolve_path=True)
)
@click.option(
    '--divisor', type=float, default=1, help='An optional number, that the summed '
    'row will be divided by. For example, this can be a timestep, which can be used '
    'to ensure that a summed row of irradiance yields cumulative radiation over '
    'the entire time period of the matrix.'
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def direct_sun_hours(
    input_matrix, divisor, output_folder
):
    """Post-process a Radiance matrix to direct sun hours and cumulative direct
    sun hours.

    \b
    This command will convert values in the Radiance matrix file to 0-1 values.
    The output will be a direct sun hours file, and a cumulative direct sun hours
    file where the values are the summed values for each row.

    Args:
        input-matrix: A Radiance matrix file.
    """
    array = binary_to_array(input_matrix)

    try:
        boolean_array = (array > 0) & (array <= np.inf)

        direct_sun_hours_array = boolean_array.astype(np.uint8)
        cumulative_array = direct_sun_hours_array.sum(axis=1) / divisor

        direct_sun_hours_file = Path(output_folder, 'direct_sun_hours')
        direct_sun_hours_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(direct_sun_hours_file, direct_sun_hours_array)

        cumulative_file = Path(output_folder, 'cumulative.res')
        cumulative_file.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(cumulative_file, cumulative_array, fmt='%.2f')
    except Exception:
        _logger.exception('Failed to convert the input file to direct sun hours.')
        sys.exit(1)
    else:
        sys.exit(0)
