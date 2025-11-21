"""Handlers to convert inputs that accept ladybug data collections."""
import os

from ladybug.datacollection import HourlyContinuousCollection, \
    HourlyDiscontinuousCollection
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval
from honeybee_energy.lib.schedules import schedule_by_identifier

from .helper import get_tempfile, write_sch_values_to_csv


def schedule_to_csv(value):
    """Translate a honeybee schedule or data collection into a CSV for annual daylight.

    Args:
        value: An annual occupancy schedule, either as a path to a csv file,
            a Ladybug Hourly Continuous Data Collection or a HB-Energy
            schedule object. This can also be the identifier of a schedule in
            your HB-Energy schedule library. Any value in this schedule that is
            0.1 or above will be considered occupied.

    Returns:
        str -- Path to a CSV of an annual schedule file. Values are 0-1 separated
            by new line.
    """
    if isinstance(value, str):
        if not os.path.isfile(value):  # check if it's already a CSV file
            try:  # the only other acceptable string is a schedule identifier
                sch = schedule_by_identifier(value)
                sv = sch.values() if isinstance(sch, ScheduleRuleset) else sch.values
                value = write_sch_values_to_csv(get_tempfile('csv', sch.identifier), sv)
            except ValueError:  # not found in the schedule library
                raise ValueError(
                    '"{}" is not a path to a CSV file or a schedule in the honeybee-'
                    'energy schedule library.'.format(value)
                )
    elif isinstance(value, ScheduleRuleset):
        value = write_sch_values_to_csv(
            get_tempfile('csv', value.identifier), value.values())
    elif isinstance(value, ScheduleFixedInterval):
        value = write_sch_values_to_csv(
            get_tempfile('csv', value.identifier), value.values)
    elif isinstance(value, HourlyContinuousCollection):
        value = write_sch_values_to_csv(get_tempfile('csv', 'occupancy'), value.values)
    else:
        raise ValueError(
            'Excpected a path to a CSV, an hourly data collection, or a honeybee '
            'schedule. Not {}.'.format(type(value))
        )
    return value


def data_to_csv(value):
    """Translate a data collection into a CSV.

    Args:
        value: An occupancy schedule, either as a path to a csv file,
            a Ladybug Hourly Data Collection.

    Returns:
        str -- Path to a CSV of an annual schedule file. Values are 0-1 separated
            by new line.
    """
    if isinstance(value, str):
        assert os.path.isfile(value), \
            'Failed to find CSV schedule file at: {}'.format(value)
    elif isinstance(value, (HourlyContinuousCollection, HourlyDiscontinuousCollection)):
        value = write_sch_values_to_csv(get_tempfile('csv', 'occupancy'), value.values)
    else:
        raise ValueError(
            'Excpected a path to a CSV or an hourly data collection. '
            'Not {}.'.format(type(value))
        )
    return value
