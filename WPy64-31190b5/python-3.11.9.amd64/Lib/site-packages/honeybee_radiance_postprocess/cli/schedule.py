"""honeybee radiance postprocess schedule commands."""
import click
import sys
import logging

from ..results.annual_daylight import AnnualDaylight
from ..dynamic import DynamicSchedule

_logger = logging.getLogger(__name__)


@click.group(help='Commands to create schedules for Radiance results.')
def schedule():
    pass


@schedule.command('control-schedules')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--base-schedule-file', '-bs', help='Path to a schedule file. A list of '
    '8760 fractional values for the lighting schedule representing the usage '
    'of lights without any daylight controls. The values of this schedule '
    'will be multiplied by the hourly dimming fraction to yield the output '
    'lighting schedules. If None, a schedule from 9AM to 5PM on weekdays will '
    'be used.',
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
    '--ill-setpoint',
    help='A number for the illuminance setpoint in lux beyond which electric '
    'lights are dimmed if there is sufficient daylight.', default=300,
    type=float, show_default=True
)
@click.option(
    '--min-power-in',
    help='A number between 0 and 1 for the the lowest power the lighting '
    'system can dim down to, expressed as a fraction of maximum input power.',
    default=0.3, type=float, show_default=True
)
@click.option(
    '--min-light-out',
    help='A number between 0 and 1 the lowest lighting output the lighting '
    'system can dim down to, expressed as a fraction of maximum light output. '
    'Note that setting this to 1 means lights are not dimmed at all until the '
    'illuminance setpoint is reached.', default=0.2, type=float,
    show_default=True
)
@click.option(
    '--off-at-min', is_flag=True, default=False, help='Boolean to '
    'note whether lights should switch off completely when they get to the '
    'minimum power input.'
)
@click.option(
    '--sub-folder', '-sf', help='Optional relative path for subfolder to '
    'write output schedule files.', default='schedules'
)
def control_schedules(
    folder, states, grids_filter, base_schedule_file, ill_setpoint,
    min_power_in, min_light_out, off_at_min, sub_folder
):
    """Generate electric lighting schedules from annual daylight results.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. Folder should include grids_info.json and sun-up-hours.txt. The
            command uses the list in grids_info.json to find the result files for each
            sensor grid.
    """
    try:
        if base_schedule_file:
            with open(base_schedule_file) as base_schedule:
                base_schedule = [float(h) for h in base_schedule.readlines()]
        else:
            base_schedule = None

        if states:
            states = DynamicSchedule.from_json(states)

        results = AnnualDaylight(folder)
        results.daylight_control_schedules_to_folder(
            sub_folder, states=states, grids_filter=grids_filter,
            base_schedule=base_schedule, ill_setpoint=ill_setpoint,
            min_power_in=min_power_in, min_light_out=min_light_out,
            off_at_min=off_at_min)
    except Exception:
        _logger.exception('Failed to generate control schedules.')
        sys.exit(1)
    else:
        sys.exit(0)
