# coding=utf-8
"""honeybee-doe2 schedule translators."""
from __future__ import division
import re

from ladybug.dt import Date, MONTHNAMES
from ladybug.analysisperiod import AnalysisPeriod
from honeybee.typing import clean_doe2_string, clean_ep_string
from honeybee_energy.schedule.day import ScheduleDay
from honeybee_energy.schedule.rule import ScheduleRule
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.lib.scheduletypelimits import fractional, on_off, temperature

from .config import RES_CHARS
from .util import generate_inp_string, generate_inp_string_list_format, \
    clean_inp_file_contents, parse_inp_string


"""____________TRANSLATORS FROM HONEYBEE TO INP____________"""


def schedule_type_limit_to_inp(type_limit):
    """Get the DOE-2 type for a honeybee-energy ScheduleTypeLimit."""
    if type_limit is None:
        return 'FRACTION'
    elif type_limit.display_name  == 'Multiplier':
        return "MULTIPLIER"
    elif type_limit.unit_type == 'Temperature':
        return 'TEMPERATURE'
    else:
        if type_limit.numeric_type == 'Discrete':
            return 'ON/OFF'
        else:
            return 'FRACTION'


def schedule_day_to_inp(day_schedule, type_limit=None):
    """Convert a ScheduleDay into a DAY-SCHEDULE INP string."""
    # setup the DOE-2 identifier and lists for keywords and values
    doe2_id = clean_doe2_string(day_schedule.identifier, RES_CHARS)
    type_text = schedule_type_limit_to_inp(type_limit)
    keywords, values = ['TYPE'], [type_text]
    hour_values = day_schedule.values_at_timestep(1)

    # convert temperature to fahrenheit if the type if temperature
    if type_text == 'TEMPERATURE':
        hour_values = [round(v * (9. / 5.) + 32., 2) for v in hour_values]

    # setup a function to format list of values correctly
    def _format_day_values(values_to_format):
        if len(values_to_format) == 1:
            return '({})'.format(round(values_to_format[0], 3))
        elif len(values_to_format) < 5:
            return str(tuple(round(v, 3) for v in values_to_format))
        else:  # we have to format it with multiple lines
            spc = ' ' * 31
            full_str = '('
            for i, v in enumerate(values_to_format):
                if i == len(values_to_format) - 1:
                    full_str += str(round(v, 3)) + ')'
                elif (i + 1) % 5 == 0:
                    full_str += str(round(v, 3)) + ',\n' + spc
                else:
                    full_str += str(round(v, 3)) + ', '
            return full_str

    # loop through the hourly values and write them in the format DOE-2 likes
    prev_count, prev_hour, prev_values = 0, 1, [hour_values[0]]
    for i, val in enumerate(hour_values):
        if i == 0:
            continue  # ignore the first value already in the list
        if val == prev_values[-1]:
            prev_count += 1
            if len(prev_values) > 1:
                keywords.append('HOURS')
                if prev_hour != i - 1:
                    values.append('({}, {})'.format(prev_hour, i - 1))
                    keywords.append('VALUES')
                    values.append(_format_day_values(prev_values[:-1]))
                    prev_values = [prev_values[-1]]
                    prev_hour = i
                else:
                    values.append('({}, {})'.format(prev_hour, i))
                    keywords.append('VALUES')
                    values.append(_format_day_values(prev_values))
                    prev_values = [prev_values[-1]]
                    prev_hour = i + 1
            continue  # wait for the value to change
        if prev_count == 0:  # just keep assembling the list of values
            prev_values.append(val)
        else:
            keywords.append('HOURS')
            values.append('({}, {})'.format(prev_hour, i))
            keywords.append('VALUES')
            values.append(_format_day_values(prev_values))
            prev_values = [val]
            prev_hour = i + 1
        prev_count = 0
    keywords.append('HOURS')
    values.append('({}, {})'.format(prev_hour, 24))
    keywords.append('VALUES')
    values.append(_format_day_values(prev_values))

    # return the INP string
    return generate_inp_string(doe2_id, 'DAY-SCHEDULE', keywords, values)


def schedule_ruleset_to_inp(schedule):
    """Convert a ScheduleRuleset into a WEEK-SCHEDULE-PD and SCHEDULE INP strings.

    Note that this method only outputs SCHEDULE and WEEK-SCHEDULE objects
    However, to write the full schedule into an INP, the schedules's
    day_schedules must also be written.

    Returns:
        A tuple with two elements

        -   year_schedule: Text string representation of the SCHEDULE
            describing this schedule.

        -   week_schedules: A list of WEEK-SCHEDULE-PD text strings that are
            referenced in the year_schedule.
    """
    # setup the DOE-2 identifier and lists for keywords and values
    doe2_id = clean_doe2_string(schedule.identifier, RES_CHARS)
    type_text = schedule_type_limit_to_inp(schedule.schedule_type_limit)
    day_types = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
        'Sunday', 'Holiday', 'Winter Design Day', 'Summer Design Day'
    ]

    def _get_week_list(schedule, rule_indices):
        """Get a list of the ScheduleDay identifiers applied on each day of the week."""
        week_list = []
        for dow in range(7):
            for i in rule_indices:
                if schedule._schedule_rules[i].week_apply_tuple[dow]:
                    day_sch_id = schedule._schedule_rules[i].schedule_day.identifier
                    week_list.append(clean_doe2_string(day_sch_id, RES_CHARS))
                    break
            else:  # no rule applies; use default_day_schedule.
                day_sch_id = schedule.default_day_schedule.identifier
                week_list.append(clean_doe2_string(day_sch_id, RES_CHARS))
        week_list.append(week_list.pop(0))  # DOE-2 starts week on Monday; not Sunday
        return week_list

    def _get_extra_week_fields(schedule):
        """Get schedule identifiers of extra days in Schedule:Week."""
        # add summer and winter design days
        week_fields = []
        if schedule._holiday_schedule is not None:
            day_sch_id = schedule._holiday_schedule.identifier
            week_fields.append(clean_doe2_string(day_sch_id, RES_CHARS))
        else:
            day_sch_id = schedule._default_day_schedule.identifier
            week_fields.append(clean_doe2_string(day_sch_id, RES_CHARS))
        if schedule._winter_designday_schedule is not None:
            day_sch_id = schedule._winter_designday_schedule.identifier
            week_fields.append(clean_doe2_string(day_sch_id, RES_CHARS))
        else:
            day_sch_id = schedule._default_day_schedule.identifier
            week_fields.append(clean_doe2_string(day_sch_id, RES_CHARS))
        if schedule._summer_designday_schedule is not None:
            day_sch_id = schedule._summer_designday_schedule.identifier
            week_fields.append(clean_doe2_string(day_sch_id, RES_CHARS))
        else:
            day_sch_id = schedule._default_day_schedule.identifier
            week_fields.append(clean_doe2_string(day_sch_id, RES_CHARS))
        return week_fields

    def _inp_week_schedule_from_rule_indices(schedule, rule_indices, week_index):
        """Create an INP string of a week schedule from a list of rules indices."""
        week_sch_id = '{}_Week {}'.format(schedule.identifier, week_index)
        week_sch_id = clean_doe2_string(week_sch_id, RES_CHARS)
        week_fields = []
        # check rules that apply for the days of the week
        week_fields.extend(_get_week_list(schedule, rule_indices))
        # add extra days (including summer and winter design days)
        week_fields.extend(_get_extra_week_fields(schedule))
        week_keywords, week_values = ['TYPE'], [type_text]
        day_list = []
        for day_type, day_sch in zip(day_types, week_fields):
            day_list.append('"{}", $ {}'.format(day_sch, day_type))
        week_keywords.append('DAY-SCHEDULES')
        week_values.append(day_list)
        week_schedule = generate_inp_string_list_format(
            week_sch_id, 'WEEK-SCHEDULE-PD', week_keywords, week_values)
        return week_schedule, week_sch_id

    def _inp_week_schedule_from_week_list(schedule, week_list, week_index):
        """Create an INP string of a week schedule from a list ScheduleDay identifiers.
        """
        week_sch_id = '{}_Week {}'.format(schedule.identifier, week_index)
        week_sch_id = clean_doe2_string(week_sch_id, RES_CHARS)
        week_fields = list(week_list)
        week_fields.append(week_fields.pop(0))  # DOE-2 starts week on Monday; not Sunday
        week_fields.extend(_get_extra_week_fields(schedule))
        week_keywords, week_values = ['TYPE'], [type_text]
        day_list = []
        for day_type, day_sch in zip(day_types, week_fields):
            day_list.append('"{}", $ {}'.format(day_sch, day_type))
        week_keywords.append('DAY-SCHEDULES')
        week_values.append(day_list)
        week_schedule = generate_inp_string_list_format(
            week_sch_id, 'WEEK-SCHEDULE-PD', week_keywords, week_values)
        return week_schedule, week_sch_id

    # prepare to create a full Schedule:Year
    week_schedules = []
    if schedule.is_single_week:  # create the only one week schedule
        wk_sch, wk_sch_id = \
            _inp_week_schedule_from_rule_indices(schedule, range(len(schedule)), 1)
        week_schedules.append(wk_sch)
        yr_wk_s_ids = [wk_sch_id]
        yr_wk_dt_range = [[Date(1, 1), Date(12, 31)]]
    else:  # create a set of week schedules throughout the year
        # loop through 365 days of the year to find unique combinations of rules
        rules_each_day = []
        for doy in range(1, 366):
            rules_on_doy = tuple(i for i, rule in enumerate(schedule._schedule_rules)
                                 if rule.does_rule_apply_doy(doy))
            rules_each_day.append(rules_on_doy)
        unique_rule_sets = set(rules_each_day)
        # check if any combination yield the same week schedule and remove duplicates
        week_tuples = [tuple(_get_week_list(schedule, rule_set))
                       for rule_set in unique_rule_sets]
        unique_week_tuples = list(set(week_tuples))
        # create the unique week schedules from the combinations of rules
        week_sched_ids = []
        for i, week_list in enumerate(unique_week_tuples):
            wk_schedule, wk_sch_id = \
                _inp_week_schedule_from_week_list(schedule, week_list, i + 1)
            week_schedules.append(wk_schedule)
            week_sched_ids.append(wk_sch_id)
        # create a dictionary mapping unique rule index lists to week schedule ids
        rule_set_map = {}
        for rule_i, week_list in zip(unique_rule_sets, week_tuples):
            unique_week_i = unique_week_tuples.index(week_list)
            rule_set_map[rule_i] = week_sched_ids[unique_week_i]
        # loop through all 365 days of the year to find when rules change
        yr_wk_s_ids = []
        yr_wk_dt_range = []
        prev_week_sched = None
        for doy in range(1, 366):
            week_sched = rule_set_map[rules_each_day[doy - 1]]
            if week_sched != prev_week_sched:  # change to a new rule set
                yr_wk_s_ids.append(week_sched)
                if doy != 1:
                    yr_wk_dt_range[-1].append(Date.from_doy(doy - 1))
                    yr_wk_dt_range.append([Date.from_doy(doy)])
                else:
                    yr_wk_dt_range.append([Date(1, 1)])
                prev_week_sched = week_sched
        yr_wk_dt_range[-1].append(Date(12, 31))

    # create the year fields
    keywords, values = ['TYPE'], [type_text]
    for i, (wk_sch_id, dt_range) in enumerate(zip(yr_wk_s_ids, yr_wk_dt_range)):
        end_date = dt_range[1]
        thru = 'THRU {} {}'.format(MONTHNAMES[end_date.month - 1].upper(), end_date.day)
        keywords.append(thru)
        values.append('"{}"'.format(wk_sch_id))
    year_schedule = generate_inp_string(doe2_id, 'SCHEDULE', keywords, values)
    return year_schedule, week_schedules


def schedule_fixed_interval_to_inp(schedule):
    """Convert a ScheduleFixedInterval to INP strings.

    Note that true Fixed Interval schedules are not supported by DOE-2 and there
    is no way to faithfully translate them given that DOE-2 SCHEDULE objects have
    a hard limit of 12 THRU statements. This method tries to write as best of
    an approximation for the schedule as possible by averaging the hourly values
    from each day of the fixed interval schedule. A separate day schedule will
    be used for each month in an attempt to account for changes in the fixed
    interval schedule over the year.

    All of this will allow the translation to succeed and gives roughly matching
    behavior of the DOE-2 simulation to the EnergyPlus simulation. However, it is
    recommended that users replace ScheduleFixedIntervals with ScheduleRulesets
    that they know best represents the schedule. Or EnergyPlus should be used for
    the simulation instead of DOE-2.

    Returns:
        A tuple with three elements

        -   year_schedule: Text string representation of the SCHEDULE
            describing this schedule.

        -   week_schedules: A list of WEEK-SCHEDULE text strings that are
            referenced in the year_schedule.

        -   day_schedules: A list of DAY-SCHEDULE-PD text strings that are
            referenced in the week_schedules.
    """
    # setup the DOE-2 identifier and lists for keywords and values
    doe2_id = clean_doe2_string(schedule.identifier, RES_CHARS)
    base_id = clean_doe2_string(schedule.identifier, RES_CHARS - 6)
    type_text = schedule_type_limit_to_inp(schedule.schedule_type_limit)

    # loop through the months of the year and create appropriate schedules
    day_schedules, week_schedules = [], []
    year_keywords, year_values = ['TYPE'], [type_text]
    sch_data = schedule.data_collection
    if sch_data.header.analysis_period.timestep != 1:
        sch_data = sch_data.cull_to_timestep(1)
    for month_i in range(1, 13):
        # create the day schedules
        month_name = AnalysisPeriod.MONTHNAMES[month_i]
        month_days = AnalysisPeriod.NUMOFDAYSEACHMONTH[month_i - 1]
        week_id = '{}{}'.format(base_id, month_name)
        day_id = '{}{}'.format(week_id, 'Day')
        period = AnalysisPeriod(st_month=month_i, end_month=month_i, end_day=month_days)
        month_data = sch_data.filter_by_analysis_period(period)
        mon_per_hr = month_data.average_monthly_per_hour()
        hour_values = [round(v, 3) for v in mon_per_hr.values]
        if type_text == 'TEMPERATURE':
            hour_values = [round(v * (9. / 5.) + 32., 2) for v in hour_values]
        day_keywords, day_values = ['TYPE', 'VALUES'], [type_text, hour_values]
        day_inp_str = generate_inp_string_list_format(
            day_id, 'DAY-SCHEDULE-PD', day_keywords, day_values)
        day_schedules.append(day_inp_str)
        # create week schedule
        week_keywords = ['TYPE', 'DAYS', 'DAY-SCHEDULES']
        week_values = [type_text, '(ALL)', '("{}")'.format(day_id)]
        week_sch = generate_inp_string(
            week_id, 'WEEK-SCHEDULE', week_keywords, week_values)
        week_schedules.append(week_sch)
        # add values to the year schedules
        thru = 'THRU {} {}'.format(month_name.upper(), period.end_day)
        year_keywords.append(thru)
        year_values.append('"{}"'.format(week_id))

    # return all of the strings
    year_schedule = generate_inp_string(doe2_id, 'SCHEDULE', year_keywords, year_values)
    return year_schedule, week_schedules, day_schedules


"""____________TRANSLATORS FROM INP TO HONEYBEE____________"""


def schedule_type_limit_from_inp(inp_type_string):
    """Get a honeybee-energy ScheduleTypeLimit for a given DOE-2 schedule type."""
    clean_type = inp_type_string.strip().upper()
    if clean_type == 'ON/OFF':
        return on_off
    elif clean_type == 'TEMPERATURE':
        return temperature
    else:
        return fractional


def schedule_day_from_inp(inp_string):
    """Create a Honeybee ScheduleDay from a DOE-2 INP text string.

    Note that this method can accept both types of DOE-2 Day Schedules
    (DAY-SCHEDULE, DAY-SCHEDULE-PD).

    Args:
        inp_string: A text string fully describing a DOE-2 DAY-SCHEDULE.
    """
    # parse the string into properties
    u_name, command, keywords, values = parse_inp_string(inp_string)
    # extract the hourly values of the schedule
    hour_vals, sch_type = [], 'FRACTIONAL'
    if command.upper() == 'DAY-SCHEDULE-PD':
        field_dict = {k: v for k, v in zip(keywords, values)}
        sch_type = field_dict['TYPE'].upper()
        hour_vals_init = eval(field_dict['VALUES'].replace('&D', '"&D"'), {})
        if isinstance(hour_vals_init, tuple):
            for val in hour_vals_init:
                if val == '&D':
                    hour_vals.append(hour_vals[-1])
                else:
                    hour_vals.append(float(val))
            if len(hour_vals) < 24:
                for _ in range(24 - len(hour_vals)):
                    hour_vals.append(hour_vals[-1])
        else:  # a constant schedule
            hour_vals = [hour_vals_init] * 24
    elif command.upper() == 'DAY-SCHEDULE':
        prev_count = 0
        for key, val in zip(keywords, values):
            if key == 'HOURS':
                hr_range = eval(val, {})
                prev_count = hr_range[-1] - hr_range[0] + 1
            elif key == 'VALUES':
                hr_vals = eval(val, {})
                if isinstance(hr_vals, tuple):
                    hour_vals.extend(hr_vals)
                else:
                    hour_vals.extend([hr_vals] * prev_count)
            elif key == 'TYPE':
                sch_type = val.upper()
    else:
        raise ValueError('Schedule type "{}" is not recognized.'.format(command))
    # convert temperature values from F to C if type is TEMPERATURE
    if sch_type == 'TEMPERATURE':
        hour_vals = [round((v - 32.) * (5. / 9.), 2) for v in hour_vals]
    return ScheduleDay.from_values_at_timestep(clean_ep_string(u_name), hour_vals)


def _inp_day_schedule_dictionary(day_inp_strings):
    """Get a dictionary of DaySchedule objects from an INP string list."""
    day_schedule_dict = {}
    for sch_str in day_inp_strings:
        sch_str = sch_str.strip()
        try:
            sch_obj = schedule_day_from_inp(sch_str)
            day_schedule_dict[sch_obj.identifier] = sch_obj
        except Exception:
            pass  # not a schedule that can be converted
    return day_schedule_dict


def extract_all_rules_from_inp_schedule_week(
        week_inp_string, day_schedule_dict, start_date=None, end_date=None):
    """Extract all ScheduleRule objects from an INP string of a WEEK-SCHEDULE-PD.

    Args:
        week_inp_string: A text string fully describing a DOE-2 WEEK-SCHEDULE-PD.
        day_schedule_dict: A dictionary with the identifiers of ScheduleDay objects
            as keys and the corresponding ScheduleDay objects as values. These objects
            will be used to build the ScheduleRules using the week_idf_string.
        start_date: A ladybug Date object for the start of the period over which
            the ScheduleRules apply. If None, Jan 1 will be used.
        end_date: A ladybug Date object for the end of the period over which
            the ScheduleRules apply. If None, Dec 31 will be used.

    Returns:
        A tuple with five elements

        -   u_name: The unique name of the WEEK-SCHEDULE-PD.

        -   schedule_rules: A list of ScheduleRule objects that together describe
            the WEEK-SCHEDULE-PD.

        -   holiday: Text for the name of the SCHEDULE-DAY for the Holiday.

        -   winter_dd: Text for the name of the SCHEDULE-DAY for the Winter Design Day.

        -   summer_dd: Text for the name of the SCHEDULE-DAY for the Summer Design Day.
    """
    # parse the string into properties
    u_name, command, keywords, values = parse_inp_string(week_inp_string)
    assert command.upper() == 'WEEK-SCHEDULE-PD', 'Week schedule must be in ' \
        'WEEK-SCHEDULE-PD format. Got "{}".'.format(command)
    # create the ScheduleRule objects from the parsed properties
    schedule_rules = []
    field_dict = {k: v for k, v in zip(keywords, values)}
    week_vals = eval(field_dict['DAY-SCHEDULES'].replace('&D', '"&D"'), {})
    applied_day_ids, prev_day = [], None
    for i, day_sch_id in enumerate(week_vals[:7]):
        day_sch_id = day_sch_id.replace('"', '')
        day_sch_id = prev_day if day_sch_id == '&D' else day_sch_id
        prev_day = day_sch_id  # increment it for the next item
        if day_sch_id not in applied_day_ids:  # make a new rule
            rule = ScheduleRule(day_schedule_dict[clean_ep_string(day_sch_id)],
                                start_date=start_date, end_date=end_date)
            if i == 6:
                rule.apply_day_by_dow(1)
            else:
                rule.apply_day_by_dow(i + 2)
            schedule_rules.append(rule)
            applied_day_ids.append(day_sch_id)
        else:  # edit one of the existing rules to apply it to the new day
            sch_rule_index = applied_day_ids.index(day_sch_id)
            rule = schedule_rules[sch_rule_index]
            if i == 6:
                rule.apply_day_by_dow(1)
            else:
                rule.apply_day_by_dow(i + 2)
    # process any of the specified Holiday or Design Day schedules
    holiday = week_vals[7] if len(week_vals) > 7 and week_vals[7] != '&D' \
        else prev_day
    winter_dd = week_vals[8] if len(week_vals) > 8 and week_vals[8] != '&D' \
        else holiday
    summer_dd = week_vals[9] if len(week_vals) > 9 and week_vals[9] != '&D' \
        else winter_dd
    return u_name, schedule_rules, holiday, winter_dd, summer_dd


def _inp_week_schedule_dictionary(week_inp_strings, day_sch_dict):
    """Get a dictionary of ScheduleRule objects from WEEK-SCHEDULE-PD strings."""
    week_schedule_dict = {}
    week_designday_dict = {}
    for sch_str in week_inp_strings:
        sch_str = sch_str.strip()
        try:
            u_name, rules, holiday, winter_dd, summer_dd = \
                extract_all_rules_from_inp_schedule_week(sch_str, day_sch_dict)
            week_schedule_dict[u_name] = rules
            week_designday_dict[u_name] = [
                day_sch_dict[clean_ep_string(holiday)],
                day_sch_dict[clean_ep_string(summer_dd)],
                day_sch_dict[clean_ep_string(winter_dd)]
            ]
        except Exception:
            pass  # schedule is not translate-able
    return week_schedule_dict, week_designday_dict


def _convert_schedule_year(year_inp_string, week_sch_dict, week_dd_dict):
    """Convert an INP string of a year SCHEDULE or SCHEDULE-PD to a ScheduleRuleset.

    Args:
        year_inp_string: An INP text string describing a DOE-2 SCHEDULE or SCHEDULE-PD.
        week_sch_dict: A dictionary of ScheduleRules from _inp_week_schedule_dictionary.
        week_dd_dict: A dictionary of design day ScheduleDay output from the
            _inp_week_schedule_dictionary method.
    """
    # use the year schedule to bring it all together
    u_name, command, keywords, values = parse_inp_string(year_inp_string)
    field_dict = {k: v for k, v in zip(keywords, values)}
    schedule_type = schedule_type_limit_from_inp(field_dict['TYPE'])
    all_rules = []
    if command.upper() == 'SCHEDULE-PD':
        week_vals = eval(field_dict['WEEK-SCHEDULES'], {})
        if not isinstance(week_vals, tuple):  # only one week for the whole year
            week_id = week_vals.replace('"', '')
            rules = week_sch_dict[week_id]
            all_rules.extend(rules)
        else:
            month_vals = eval(field_dict['MONTH'], {})
            day_vals = eval(field_dict['DAY'], {})
            prev_month, prev_day = 1, 1
            for month, day, week in zip(month_vals, day_vals, week_vals):
                week_id = week.replace('"', '')
                rules = week_sch_dict[week_id]
                st_date = Date(int(prev_month), int(prev_day))
                end_date = Date(int(month), int(day))
                for rule in rules:
                    rule.start_date = st_date
                    rule.end_date = end_date
                all_rules.extend(rules)
                end_doy = end_date.doy + 1 if end_date.doy != 365 else 365
                next_day = Date.from_doy(end_doy)
                prev_month, prev_day = next_day.month, next_day.day
    elif command.upper() == 'SCHEDULE':
        prev_month, prev_day = 1, 1
        for key, val in zip(keywords, values):
            if key.startswith('THRU'):
                week_id = val.replace('"', '')
                rules = week_sch_dict[week_id]
                st_date = Date(int(prev_month), int(prev_day))
                date_vals = key.replace('THRU ', '').split(' ')
                date_str = '{} {}'.format(date_vals[1], date_vals[0].title())
                end_date = Date.from_date_string(date_str)
                for rule in rules:
                    rule.start_date = st_date
                    rule.end_date = end_date
                all_rules.extend(rules)
                end_doy = end_date.doy + 1 if end_date.doy != 365 else 365
                next_day = Date.from_doy(end_doy)
                prev_month, prev_day = next_day.month, next_day.day

    # check to be sure the schedule days don't already have a parent
    for rule in all_rules:
        if rule.schedule_day._parent is not None:
            rule.schedule_day = rule.schedule_day.duplicate()
    default_day_schedule = all_rules[0].schedule_day
    holiday_sch, summer_dd_sch, winter_dd_sch = week_dd_dict[week_id]
    if holiday_sch._parent is not None:
        holiday_sch = holiday_sch.duplicate()
    if summer_dd_sch._parent is not None:
        summer_dd_sch = summer_dd_sch.duplicate()
    if winter_dd_sch._parent is not None:
        winter_dd_sch = summer_dd_sch.duplicate()

    # create the ScheduleRuleset and apply the design days
    sched = ScheduleRuleset(clean_ep_string(u_name), default_day_schedule,
                            all_rules[1:], schedule_type)
    ScheduleRuleset._apply_designdays_with_check(
        sched, holiday_sch, summer_dd_sch, winter_dd_sch)
    return sched


def schedule_ruleset_from_inp(year_inp_string, week_inp_strings, day_inp_strings):
    """Create a ScheduleRuleset from a DOE-2 INP text strings.

    Args:
        year_inp_string: An INP text string describing a DOE-2 SCHEDULE or SCHEDULE-PD.
        week_inp_strings: A list of INP text strings for all of the WEEK-SCHEDULE-PD
            objects used in the SCHEDULE.
        day_inp_strings: A list of text strings for all of the DAY-SCHEDULE or
            DAY-SCHEDULE-PD objects used in the week_inp_strings.
    """
    # process the schedule components
    day_schedule_dict = _inp_day_schedule_dictionary(day_inp_strings)
    week_sch_dict, week_dd_dict = _inp_week_schedule_dictionary(
        week_inp_strings, day_schedule_dict)

    # convert the year_inp_string into a ScheduleRuleset
    return _convert_schedule_year(year_inp_string, week_sch_dict, week_dd_dict)


def extract_all_schedule_ruleset_from_inp_file(inp_file):
    """Extract all ScheduleRuleset objects from a DOE-2 INP file.

    Args:
        inp_file: A path to an INP file containing objects for SCHEDULE
            (or SCHEDULE-PD) and corresponding WEEK-SCHEDULE-PD and DAY-SCHEDULE
            (or DAY-SCHEDULE-PD) objects. The SCHEDULE will be used to assemble
            all of these into a ScheduleRuleset.

    Returns:
        schedules -- A list of all Schedule objects in the INP file as
        honeybee_energy ScheduleRuleset objects.
    """
    # read the file and remove lines of comments
    file_contents = clean_inp_file_contents(inp_file)
    # extract all of the DAY-SCHEDULE objects
    day_pattern1 = re.compile(r'(?i)(".*=.*DAY-SCHEDULE\n[\s\S]*?\.\.)')
    day_pattern2 = re.compile(r'(?i)(".*=.*DAY-SCHEDULE-PD\n[\s\S]*?\.\.)')
    day_sch_str = day_pattern1.findall(file_contents) + \
        day_pattern2.findall(file_contents)
    day_schedule_dict = _inp_day_schedule_dictionary(day_sch_str)
    # extract all of the WEEK-SCHEDULE-PD objects
    week_pattern = re.compile(r'(?i)(".*=.*WEEK-SCHEDULE-PD\n[\s\S]*?\.\.)')
    week_sch_str = week_pattern.findall(file_contents)
    week_sch_dict, week_dd_dict = _inp_week_schedule_dictionary(
        week_sch_str, day_schedule_dict)
    # extract all of the SCHEDULE objects and convert to ScheduleRuleset
    year_pattern1 = re.compile(r'(?i)(".*=.*SCHEDULE\n[\s\S]*?\.\.)')
    year_pattern2 = re.compile(r'(?i)(".*=.*SCHEDULE-PD\n[\s\S]*?\.\.)')
    year_sch_str = year_pattern1.findall(file_contents) + \
        year_pattern2.findall(file_contents)

    # translate each SCHEDULE and check to be sure ScheduleDay objects are unique
    schedules = []
    for year_sch in year_sch_str:
        try:
            yr_sch = _convert_schedule_year(year_sch, week_sch_dict, week_dd_dict)
            schedules.append(yr_sch)
        except Exception:
            pass  # schedule is not translate-able
    return schedules


"""______EXTRA UTILITY FUNCTIONS RELATED TO SCHEDULES______"""


def energy_trans_sch_to_transmittance(shade_obj):
    """Try to extract the transmittance from the shade energy properties."""
    trans = 0
    trans_sch = shade_obj.properties.energy.transmittance_schedule
    if trans_sch is not None:
        if trans_sch.is_constant:
            try:  # assume ScheduleRuleset
                trans = trans_sch.default_day_schedule[0]
            except AttributeError:  # ScheduleFixedInterval
                trans = trans_sch.values[0]
        else:  # not a constant schedule; use the average transmittance
            try:  # assume ScheduleRuleset
                sch_vals = trans_sch.values()
            except Exception:  # ScheduleFixedInterval
                sch_vals = trans_sch.values
            trans = round(sum(sch_vals) / len(sch_vals), 3)
    return trans
