# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/schedules/create.rb
"""
from __future__ import division

from honeybee_openstudio.openstudio import openstudio, openstudio_model, os_vector_len


def create_constant_schedule_ruleset(model, value, name=None, schedule_type_limit=None):
    """Create constant ScheduleRuleset with a given value.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object
        value: [Double] the value to use, 24-7, 365
        name: [String] the name of the schedule
        schedule_type_limit: [String] the name of a schedule type limit.
            options are Dimensionless, Temperature, Humidity Ratio, Fraction,
            Fractional, OnOff, and Activity
    """
    # check to see if schedule exists with same name and constant value and return if true
    if name is not None:
        existing_sch = model.getScheduleRulesetByName(name)
        if existing_sch.is_initialized():
            existing_sch = existing_sch.get()
            existing_day_sch_vals = existing_sch.defaultDaySchedule().values()
            if os_vector_len(existing_day_sch_vals) == 1:
                if abs(existing_day_sch_vals[0] - value) < 1.0e-6:
                    return existing_sch
    # create ScheduleRuleset
    schedule = openstudio_model.ScheduleRuleset(model)
    schedule.defaultDaySchedule().addValue(openstudio.Time(0, 24, 0, 0), value)
    # set name
    if name is not None:
        schedule.setName(name)
        schedule.defaultDaySchedule().setName('{} Default'.format(name))
    # set schedule type limits
    if schedule_type_limit is not None:
        sch_type_limits_obj = create_schedule_type_limits(
            model, standard_schedule_type_limit=schedule_type_limit)
        schedule.setScheduleTypeLimits(sch_type_limits_obj)
    return schedule


def create_schedule_type_limits(
        model, standard_schedule_type_limit=None, name=None, lower_limit_value=None,
        upper_limit_value=None, numeric_type=None, unit_type=None):
    """Create a ScheduleTypeLimits object for a schedule.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        standard_schedule_type_limit: [String] the name of a standard schedule
            type limit with predefined limits. Options are Dimensionless, Temperature,
            Humidity Ratio, Fraction, Fractional, OnOff, and Activity.
        name: [String] the name of the schedule type limits.
        lower_limit_value: [double] the lower limit value for the schedule type.
        upper_limit_value: [double] the upper limit value for the schedule type.
        numeric_type: [String] the numeric type, options are Continuous or Discrete.
        unit_type: [String] the unit type, options are defined in EnergyPlus I/O reference.
    """

    if standard_schedule_type_limit is None:
        if lower_limit_value is None or upper_limit_value is None or \
                numeric_type is None or unit_type is None:
            msg = 'If calling create_schedule_type_limits without a ' \
                'standard_schedule_type_limit, you must specify all properties ' \
                'of ScheduleTypeLimits.'
            print(msg)
            return None

        schedule_type_limits = openstudio_model.scheduleTypeLimits(model)
        if name is not None:
            schedule_type_limits.setName(name)
        if lower_limit_value is not None:
            schedule_type_limits.setLowerLimitValue(lower_limit_value)
        if upper_limit_value is not None:
            schedule_type_limits.setUpperLimitValue(upper_limit_value)
        if numeric_type is not None:
            schedule_type_limits.setNumericType(numeric_type)
        if unit_type is not None:
            schedule_type_limits.setUnitType(unit_type)
    else:
        schedule_type_limits = model.getScheduleTypeLimitsByName(
            standard_schedule_type_limit)
        if not schedule_type_limits.is_initialized():
            stl_name = standard_schedule_type_limit.lower()
            if stl_name == 'dimensionless':
                schedule_type_limits = openstudio_model.ScheduleTypeLimits(model)
                schedule_type_limits.setName('Dimensionless')
                schedule_type_limits.setLowerLimitValue(0.0)
                schedule_type_limits.setUpperLimitValue(1000.0)
                schedule_type_limits.setNumericType('Continuous')
                schedule_type_limits.setUnitType('Dimensionless')
            elif stl_name == 'temperature':
                schedule_type_limits = openstudio_model.ScheduleTypeLimits(model)
                schedule_type_limits.setName('Temperature')
                schedule_type_limits.setLowerLimitValue(0.0)
                schedule_type_limits.setUpperLimitValue(100.0)
                schedule_type_limits.setNumericType('Continuous')
                schedule_type_limits.setUnitType('Temperature')
            elif stl_name == 'humidity ratio':
                schedule_type_limits = openstudio_model.ScheduleTypeLimits(model)
                schedule_type_limits.setName('Humidity Ratio')
                schedule_type_limits.setLowerLimitValue(0.0)
                schedule_type_limits.setUpperLimitValue(0.3)
                schedule_type_limits.setNumericType('Continuous')
                schedule_type_limits.setUnitType('Dimensionless')
            elif stl_name in ('fraction', 'fractional'):
                schedule_type_limits = openstudio_model.ScheduleTypeLimits(model)
                schedule_type_limits.setName('Fraction')
                schedule_type_limits.setLowerLimitValue(0.0)
                schedule_type_limits.setUpperLimitValue(1.0)
                schedule_type_limits.setNumericType('Continuous')
                schedule_type_limits.setUnitType('Dimensionless')
            elif stl_name == 'onoff':
                schedule_type_limits = openstudio_model.ScheduleTypeLimits(model)
                schedule_type_limits.setName('OnOff')
                schedule_type_limits.setLowerLimitValue(0)
                schedule_type_limits.setUpperLimitValue(1)
                schedule_type_limits.setNumericType('Discrete')
                schedule_type_limits.setUnitType('Availability')
            elif stl_name == 'activity':
                schedule_type_limits = openstudio_model.ScheduleTypeLimits(model)
                schedule_type_limits.setName('Activity')
                schedule_type_limits.setLowerLimitValue(70.0)
                schedule_type_limits.setUpperLimitValue(1000.0)
                schedule_type_limits.setNumericType('Continuous')
                schedule_type_limits.setUnitType('ActivityLevel')
            else:
                msg = 'Invalid standard_schedule_type_limit for method ' \
                    'create_schedule_type_limits.'
                raise ValueError(msg)
        else:
            schedule_type_limits = schedule_type_limits.get()
    return schedule_type_limits


def model_add_schedule(model, schedule_name):
    """Get a schedule from the Model or always on schedule if it does not exist."""
    if schedule_name is None or schedule_name == '':
        return model.alwaysOnDiscreteSchedule()
    schedule = model.getScheduleByName(schedule_name)
    if schedule.is_initialized():
        return schedule.get()
    return model.alwaysOnDiscreteSchedule()
