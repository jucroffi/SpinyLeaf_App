# coding=utf-8
"""Utilities to convert schedule dictionaries to Python objects."""
from __future__ import division
import os

from ladybug.futil import write_to_file
from ladybug.dt import Date, Time
from ladybug.analysisperiod import AnalysisPeriod
from honeybee.altnumber import no_limit
from honeybee.typing import clean_ep_string
from honeybee_energy.schedule.typelimit import ScheduleTypeLimit
from honeybee_energy.schedule.day import ScheduleDay
from honeybee_energy.schedule.rule import ScheduleRule
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval

from honeybee_openstudio.openstudio import OSScheduleTypeLimits, OSScheduleRuleset, \
    OSScheduleRule, OSScheduleDay, OSScheduleFixedInterval, OSExternalFile, \
    OSScheduleFile, OSVector, OSTime, OSTimeSeries


"""____________TRANSLATORS TO OPENSTUDIO____________"""


def schedule_type_limits_to_openstudio(type_limit, os_model):
    """Convert Honeybee ScheduleTypeLimit to OpenStudio ScheduleTypeLimits."""
    os_type_limit = OSScheduleTypeLimits(os_model)
    os_type_limit.setName(type_limit.identifier)
    if type_limit._display_name is not None:
        os_type_limit.setDisplayName(type_limit.display_name)
    if type_limit.lower_limit != no_limit:
        os_type_limit.setLowerLimitValue(type_limit.lower_limit)
    if type_limit.upper_limit != no_limit:
        os_type_limit.setUpperLimitValue(type_limit.upper_limit)
    os_type_limit.setNumericType(type_limit.numeric_type)
    os_type_limit.setUnitType(type_limit.unit_type)
    return os_type_limit


def schedule_day_to_openstudio(schedule_day, os_model):
    """Convert Honeybee ScheduleDay to OpenStudio ScheduleDay."""
    os_day_sch = OSScheduleDay(os_model)
    os_day_sch.setName(schedule_day.identifier)
    if schedule_day._display_name is not None:
        os_day_sch.setDisplayName(schedule_day.display_name)
    values_day = schedule_day.values
    times_day = [tm.to_array() for tm in schedule_day.times]
    times_day.pop(0)  # Remove [0, 0] from array at index 0.
    times_day.append((24, 0))  # Add [24, 0] at index 0
    for i, val in enumerate(values_day):
        time_until = OSTime(0, times_day[i][0], times_day[i][1], 0)
        os_day_sch.addValue(time_until, val)
    return os_day_sch


def schedule_ruleset_to_openstudio(schedule, os_model):
    """Convert Honeybee ScheduleRuleset to OpenStudio ScheduleRuleset."""
    # create openstudio schedule ruleset object
    os_sch_ruleset = OSScheduleRuleset(os_model)
    os_sch_ruleset.setName(schedule.identifier)
    if schedule._display_name is not None:
        os_sch_ruleset.setDisplayName(schedule.display_name)
    # assign schedule type limit
    os_type_limit = None
    if schedule.schedule_type_limit:
        os_type_limit_ref = os_model.getScheduleTypeLimitsByName(
            schedule.schedule_type_limit.identifier)
        if os_type_limit_ref.is_initialized():
            os_type_limit = os_type_limit_ref.get()
            os_sch_ruleset.setScheduleTypeLimits(os_type_limit)
    # loop through day schedules and create openstudio schedule day objects
    day_schs = {}
    def_day = schedule.default_day_schedule
    for day_sch in schedule.day_schedules:
        if day_sch.identifier != def_day.identifier:
            os_day_sch = schedule_day_to_openstudio(day_sch, os_model)
            if os_type_limit is not None:
                os_day_sch.setScheduleTypeLimits(os_type_limit)
            day_schs[day_sch.identifier] = os_day_sch
    # assign default day schedule
    os_def_day_sch = os_sch_ruleset.defaultDaySchedule()
    day_schs[def_day.identifier] = os_def_day_sch
    if os_type_limit is not None:
        os_def_day_sch.setScheduleTypeLimits(os_type_limit)
    os_def_day_sch.setName(def_day.identifier)
    if def_day._display_name is not None:
        os_def_day_sch.setDisplayName(def_day.display_name)
    values_day = def_day.values
    times_day = [tm.to_array() for tm in def_day.times]
    times_day.pop(0)  # Remove [0, 0] from array at index 0.
    times_day.append((24, 0))  # Add [24, 0] at index 0
    for i, val in enumerate(values_day):
        time_until = OSTime(0, times_day[i][0], times_day[i][1], 0)
        os_def_day_sch.addValue(time_until, val)
    # assign holiday schedule
    if schedule.holiday_schedule is not None:
        holiday_schedule = day_schs[schedule.holiday_schedule.identifier]
        os_sch_ruleset.setHolidaySchedule(holiday_schedule)
    # assign summer design day schedule
    if schedule.summer_designday_schedule is not None:
        summer_design_day = day_schs[schedule.summer_designday_schedule.identifier]
        os_sch_ruleset.setSummerDesignDaySchedule(summer_design_day)
    # assign winter design day schedule
    if schedule.winter_designday_schedule is not None:
        winter_design_day = day_schs[schedule.winter_designday_schedule.identifier]
        os_sch_ruleset.setWinterDesignDaySchedule(winter_design_day)
    # assign schedule rules
    for i, rule in enumerate(schedule.schedule_rules):
        os_rule = OSScheduleRule(os_sch_ruleset)
        os_rule.setApplySunday(rule.apply_sunday)
        os_rule.setApplyMonday(rule.apply_monday)
        os_rule.setApplyTuesday(rule.apply_tuesday)
        os_rule.setApplyWednesday(rule.apply_wednesday)
        os_rule.setApplyThursday(rule.apply_thursday)
        os_rule.setApplyFriday(rule.apply_friday)
        os_rule.setApplySaturday(rule.apply_saturday)
        start_date = os_model.makeDate(rule.start_date.month, rule.start_date.day)
        end_date = os_model.makeDate(rule.end_date.month, rule.end_date.day)
        os_rule.setStartDate(start_date)
        os_rule.setEndDate(end_date)
        schedule_rule_day = day_schs[rule.schedule_day.identifier]
        values_day = schedule_rule_day.values()
        times_day = schedule_rule_day.times()
        for tim, val in zip(times_day, values_day):
            rule_day = os_rule.daySchedule()
            rule_day.addValue(tim, val)
        os_sch_ruleset.setScheduleRuleIndex(os_rule, i)
    return os_sch_ruleset


def schedule_fixed_interval_to_openstudio(schedule, os_model):
    """Convert Honeybee ScheduleFixedInterval to OpenStudio ScheduleFixedInterval."""
    # create the new schedule
    os_fi_sch = OSScheduleFixedInterval(os_model)
    os_fi_sch.setName(schedule.identifier)
    if schedule._display_name is not None:
        os_fi_sch.setDisplayName(schedule.display_name)
    # assign start date and the out of range value
    os_fi_sch.setStartMonth(1)
    os_fi_sch.setStartDay(1)
    os_fi_sch.setOutOfRangeValue(schedule.placeholder_value)
    # assign the interpolate value
    os_fi_sch.setInterpolatetoTimestep(schedule.interpolate)
    # assign the schedule type limit
    if schedule.schedule_type_limit:
        os_type_limit_ref = os_model.getScheduleTypeLimitsByName(
            schedule.schedule_type_limit.identifier)
        if os_type_limit_ref.is_initialized():
            os_type_limit = os_type_limit_ref.get()
            os_fi_sch.setScheduleTypeLimits(os_type_limit)
    # assign the timestep
    interval_length = int(60 / schedule.timestep)
    os_fi_sch.setIntervalLength(interval_length)
    os_interval_length = OSTime(0, 0, interval_length)
    # assign the values as a timeseries
    start_date = os_model.makeDate(1, 1)
    all_values = [float(val) for val in schedule.values_at_timestep(schedule.timestep)]
    series_values = OSVector(len(all_values))
    for i, val in enumerate(all_values):
        series_values[i] = val
    timeseries = OSTimeSeries(start_date, os_interval_length, series_values, '')
    os_fi_sch.setTimeSeries(timeseries)
    return os_fi_sch


def schedule_fixed_interval_to_openstudio_file(
        schedule, os_model, schedule_directory, include_datetimes=False):
    """Convert Honeybee ScheduleFixedInterval to OpenStudio ScheduleFile.

    Args:
        schedule: The Honeybee ScheduleFixedInterval to be converted.
        os_model: The OpenStudio Model to which the ScheduleFile will be added.
        schedule_directory: Text string of a path to a folder on this machine to
            which the CSV version of the file will be written.
        include_datetimes: Boolean to note whether a column of datetime objects
            should be written into the CSV alongside the data. Default is False,
            which will keep the resulting CSV lighter in file size but you may
            want to include such datetimes in order to verify that values align with
            the expected timestep. Note that the included datetimes will follow the
            EnergyPlus interpretation of aligning values to timesteps in which case
            the timestep to which the value is matched means that the value was
            utilized over all of the previous timestep.
    """
    # gather all of the data to be written into the CSV
    sched_data = [str(val) for val in schedule.values_at_timestep(schedule.timestep)]
    if include_datetimes:
        sched_a_per = AnalysisPeriod(timestep=schedule.timestep,
                                     is_leap_year=schedule.is_leap_year)
        sched_data = ('{},{}'.format(dt, val) for dt, val in
                      zip(sched_a_per.datetimes, sched_data))
    file_name = '{}.csv'.format(schedule.identifier.replace(' ', '_'))
    file_path = os.path.join(schedule_directory, file_name)
    # write the data into the file
    write_to_file(file_path, ',\n'.join(sched_data), True)
    full_path = os.path.abspath(file_path)
    # get the external file which points to the schedule csv file
    os_external_file = OSExternalFile.getExternalFile(os_model, full_path, False)
    if os_external_file.is_initialized():
        os_external_file = os_external_file.get()
    # create the schedule file
    column = 2 if include_datetimes else 1
    os_sch_file = OSScheduleFile(os_external_file, column, 0)
    os_sch_file.setName(schedule.identifier)
    if schedule._display_name is not None:
        os_sch_file.setDisplayName(schedule.display_name)
    os_sch_file.setInterpolatetoTimestep(schedule.interpolate)
    interval_length = int(60 / schedule.timestep)
    os_sch_file.setMinutesperItem(interval_length)
    # assign the schedule type limit
    if schedule.schedule_type_limit:
        os_type_limit_ref = os_model.getScheduleTypeLimitsByName(
            schedule.schedule_type_limit.identifier)
        if os_type_limit_ref.is_initialized():
            os_type_limit = os_type_limit_ref.get()
            os_sch_file.setScheduleTypeLimits(os_type_limit)
    return os_sch_file


def schedule_to_openstudio(schedule, os_model, schedule_directory=None):
    """Convert any Honeybee energy material into an OpenStudio object.

    Args:
        material: A honeybee-energy Python object of a material layer.
        os_model: The OpenStudio Model object to which the Room will be added.
        schedule_directory: An optional directory to be used to write Honeybee
            ScheduleFixedInterval objects to OpenStudio ScheduleFile objects
            instead of OpenStudio ScheduleFixedInterval, which translates to
            EnergyPlus Compact schedules.

    Returns:
        An OpenStudio object for the material.
    """
    if isinstance(schedule, ScheduleRuleset):
        return schedule_ruleset_to_openstudio(schedule, os_model)
    elif isinstance(schedule, ScheduleFixedInterval):
        if schedule_directory is None:
            return schedule_fixed_interval_to_openstudio(schedule, os_model)
        else:
            return schedule_fixed_interval_to_openstudio_file(
                schedule, os_model, schedule_directory)
    else:
        raise ValueError(
            '{} is not a recognized energy Schedule type'.format(type(schedule))
        )


"""____________TRANSLATORS FROM OPENSTUDIO____________"""


def schedule_type_limits_from_openstudio(os_type_limit):
    """Convert OpenStudio ScheduleTypeLimits to Honeybee ScheduleTypeLimit."""
    lower_limit = os_type_limit.lowerLimitValue().get() if \
        os_type_limit.lowerLimitValue().is_initialized() else no_limit
    upper_limit = os_type_limit.upperLimitValue().get() if \
        os_type_limit.upperLimitValue().is_initialized() else no_limit
    numeric_type = os_type_limit.numericType().get().title() if \
        os_type_limit.numericType().is_initialized() else 'Continuous'
    unit_type = os_type_limit.unitType().title()
    if unit_type == 'Deltatemperature':
        unit_type = 'DeltaTemperature'
    elif unit_type == 'Precipitationrate':
        unit_type = 'PrecipitationRate'
    elif unit_type == 'Convectioncoefficient':
        unit_type = 'ConvectionCoefficient'
    elif unit_type == 'Activitylevel':
        unit_type = 'ActivityLevel'
    elif unit_type == 'Controlmode':
        unit_type = 'Control'
    unit_type = unit_type if unit_type in ScheduleTypeLimit.UNIT_TYPES \
        else 'Dimensionless'
    type_limit = ScheduleTypeLimit(
        clean_ep_string(os_type_limit.nameString()), lower_limit, upper_limit,
        numeric_type, unit_type)
    if os_type_limit.displayName().is_initialized():
        type_limit.display_name = os_type_limit.displayName().get()
    return type_limit


def schedule_day_from_openstudio(os_day_schedule):
    """Convert OpenStudio ScheduleDay to Honeybee ScheduleDay."""
    values = [v for v in os_day_schedule.values()]
    times = [Time(0, 0)]
    for shc_time in os_day_schedule.times():
        times.append(Time(shc_time.hours(), shc_time.minutes()))
    times.pop(-1)
    interpolate = os_day_schedule.interpolatetoTimestep()
    day_schedule = ScheduleDay(clean_ep_string(os_day_schedule.nameString()),
                               values, times, interpolate)
    if os_day_schedule.displayName().is_initialized():
        day_schedule.display_name = os_day_schedule.displayName().get()
    return day_schedule


def _schedule_rule_from_openstudio(os_sch_rule, day_schedules):
    """Convert OpenStudio ScheduleRule to Honeybee ScheduleRule."""
    # create the ScheduleRule object
    sch_day_id = clean_ep_string(os_sch_rule.daySchedule().nameString())
    schedule_day = day_schedules[sch_day_id]
    apply_sunday = os_sch_rule.applySunday()
    apply_monday = os_sch_rule.applyMonday()
    apply_tuesday = os_sch_rule.applyTuesday()
    apply_wednesday = os_sch_rule.applyWednesday()
    apply_thursday = os_sch_rule.applyThursday()
    apply_friday = os_sch_rule.applyFriday()
    apply_saturday = os_sch_rule.applySaturday()
    sch_rule = ScheduleRule(
        schedule_day, apply_sunday, apply_monday, apply_tuesday, apply_wednesday,
        apply_thursday, apply_friday, apply_saturday)
    # assign the optional dates to the rule
    if os_sch_rule.startDate().is_initialized():
        start_date = os_sch_rule.startDate().get()
        start_date_arr = [start_date.monthOfYear().value(), start_date.dayOfMonth()]
        if start_date.isLeapYear():
            start_date_arr.append(True)
        sch_rule.start_date = Date.from_array(start_date_arr)
    if os_sch_rule.endDate().is_initialized():
        end_date = os_sch_rule.endDate().get()
        end_date_arr = [start_date.monthOfYear().value(), end_date.dayOfMonth()]
        if end_date.isLeapYear():
            end_date_arr.append(True)
        try:
            sch_rule.end_date = Date.from_array(end_date_arr)
        except ValueError:  # OpenStudio parsers messed up the date (eg. 9/31)
            end_date_arr[1] = end_date_arr[1] - 1
            sch_rule.end_date = Date.from_array(end_date_arr)
    return sch_rule


def schedule_ruleset_from_openstudio(os_schedule, type_limits=None):
    """Convert OpenStudio ScheduleRuleset to Honeybee ScheduleRuleset."""
    default_day_schedule = \
        clean_ep_string(os_schedule.defaultDaySchedule().nameString())
    summer_designday_schedule = \
        clean_ep_string(os_schedule.summerDesignDaySchedule().nameString())
    winter_designday_schedule = \
        clean_ep_string(os_schedule.winterDesignDaySchedule().nameString())
    holiday_schedule = \
        clean_ep_string(os_schedule.holidaySchedule().nameString())
    # create a list of all day schedules referenced in the Ruleset
    schedule_days = {}
    required_days = [
        os_schedule.defaultDaySchedule(),
        os_schedule.summerDesignDaySchedule(),
        os_schedule.winterDesignDaySchedule(),
        os_schedule.holidaySchedule()
    ]
    for os_day_sch in required_days:
        if os_day_sch.nameString() not in schedule_days:
            schedule_days[os_day_sch.nameString()] = \
                schedule_day_from_openstudio(os_day_sch)
    for os_rule in os_schedule.scheduleRules():
        os_day_sch = os_rule.daySchedule()
        if os_day_sch.nameString() not in schedule_days:
            schedule_days[os_day_sch.nameString()] = \
                schedule_day_from_openstudio(os_day_sch)
    # loop through the rules and add them along with their day schedules
    schedule_rules = []
    for os_rule in os_schedule.scheduleRules():
        rule = _schedule_rule_from_openstudio(os_rule, schedule_days)
        schedule_rules.append(rule)
    # get any schedule type limits if they exist
    typ_lim = None
    if type_limits is not None and os_schedule.scheduleTypeLimits().is_initialized():
        typ_lim = os_schedule.scheduleTypeLimits().get()
        try:
            typ_lim = type_limits[clean_ep_string(typ_lim.nameString())]
        except KeyError:  # type limit that could not be re-serialized
            typ_lim = None

    # create the schedule object
    schedule = ScheduleRuleset(
        clean_ep_string(os_schedule.nameString()), schedule_days[default_day_schedule],
        schedule_rules, typ_lim, schedule_days[holiday_schedule],
        schedule_days[summer_designday_schedule], schedule_days[winter_designday_schedule])
    if os_schedule.displayName().is_initialized():
        schedule.display_name = os_schedule.displayName().get()
    return schedule


def schedule_fixed_interval_from_openstudio(os_schedule, type_limits=None,
                                            is_leap_year=False):
    """Convert OpenStudio ScheduleFixedInterval to Honeybee ScheduleFixedInterval."""
    # get the start month
    start_month = os_schedule.startMonth()
    start_day = os_schedule.startDay()
    start_date = Date(start_month, start_day, True) if is_leap_year else \
        Date(start_month, start_day)
    interpolate = os_schedule.interpolatetoTimestep()
    # get any schedule type limits if they exist
    typ_lim = None
    if type_limits is not None and os_schedule.scheduleTypeLimits().is_initialized():
        typ_lim = os_schedule.scheduleTypeLimits().get()
        try:
            typ_lim = type_limits[clean_ep_string(typ_lim.nameString())]
        except KeyError:  # type limits that could not be re-serialized
            typ_lim = None
    # compute the timestep
    interval_length = os_schedule.intervalLength()
    timestep = 60 / int(interval_length)
    # get values from schedule fixed interval
    values = os_schedule.timeSeries().values()
    values = [values[i] for i in range(len(values))]
    # create the schedule object
    schedule = ScheduleFixedInterval(
        clean_ep_string(os_schedule.nameString()), values, typ_lim, timestep,
        start_date, interpolate=interpolate)
    if os_schedule.displayName().is_initialized():
        schedule.display_name = os_schedule.displayName().get()
    return schedule


def extract_all_schedules(os_model):
    """Extract all schedule objects from an OpenStudio Model.

    Args:
        os_model: The OpenStudio Model object from which schedules will be extracted.

    Returns:
        A dictionary of schedule objects with schedule identifiers as keys and
        schedule objects as values.
    """
    # first, gather all schedule type limits
    type_limits = {}
    for os_type_lim in os_model.getScheduleTypeLimitss():
        type_lim = schedule_type_limits_from_openstudio(os_type_lim)
        type_limits[type_lim.identifier] = type_lim
    # gather all of the schedule objects
    is_leap_year = os_model.isLeapYear()
    schedules = {}
    for os_schedule in os_model.getScheduleRulesets():
        schedule = schedule_ruleset_from_openstudio(os_schedule, type_limits)
        schedules[schedule.identifier] = schedule
    for os_schedule in os_model.getScheduleFixedIntervals():
        schedule = schedule_fixed_interval_from_openstudio(
            os_schedule, type_limits, is_leap_year)
        schedules[schedule.identifier] = schedule
    return schedules
