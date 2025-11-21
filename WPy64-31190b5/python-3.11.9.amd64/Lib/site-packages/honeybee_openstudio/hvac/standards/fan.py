# coding=utf-8
"""Module taken from OpenStudio-standards.

Prototype fan calculation methods that are the same regardless of fan type.
These methods are available to FanConstantVolume, FanOnOff, FanVariableVolume,
and FanZoneExhaust.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.Fan.rb
"""
from __future__ import division
import os
import json

from ladybug.datatype.pressure import Pressure

from honeybee_openstudio.openstudio import openstudio_model


def _load_json_data(json_name):
    """load the fan data from the JSON file."""
    try:
        cur_dir = os.path.dirname(__file__)
        data_file = os.path.join(cur_dir, 'data', json_name)
        with open(data_file, 'r') as inf:
            fan_data = json.load(inf)
    except IOError:
        fan_data = {}
        print('Failed to import fan JSON data.')
    return fan_data


def _fan_curve_coefficients_from_json(fan_curve_name):
    """Lookup fan curve coefficients."""
    for curve_obj in CURVE_DATA:
        if curve_obj['name'] == fan_curve_name:
            return curve_obj['coeff_1'], curve_obj['coeff_2'], \
                curve_obj['coeff_3'], curve_obj['coeff_4'], curve_obj['coeff_5']
    return None, None, None, None, None


PRESSURE = Pressure()
FAN_DATA = _load_json_data('ashrae_90_1.fans.json')['fans']
CURVE_DATA = _load_json_data('ashrae_90_1.curves.json')['curves']


def create_fan_by_name(
        model, standards_name, fan_name=None, fan_efficiency=None, pressure_rise=None,
        motor_efficiency=None, motor_in_airstream_fraction=None,
        fan_power_minimum_flow_rate_input_method=None,
        fan_power_minimum_flow_rate_fraction=None,
        system_availability_manager_coupling_mode=None,
        end_use_subcategory=None):
    """Create a fan with properties for a fan name in the standards data.

    Args:
        fan_name: [String] fan name.
        fan_efficiency: [Double] fan efficiency.
        pressure_rise: [Double] fan pressure rise in Pa.
        end_use_subcategory: [String] end use subcategory name.
    """
    # get the data from the JSON
    for fan_obj in FAN_DATA:
        if fan_obj['name'] == standards_name:
            fan_json = fan_obj
            break

    # create the fan
    if fan_json['type'] == 'ConstantVolume':
        fan = create_fan_constant_volume_from_json(
            model, fan_json, fan_name=fan_name, fan_efficiency=fan_efficiency,
            pressure_rise=pressure_rise, motor_efficiency=motor_efficiency,
            motor_in_airstream_fraction=motor_in_airstream_fraction,
            end_use_subcategory=end_use_subcategory)
    elif fan_json['type'] == 'OnOff':
        fan = create_fan_on_off_from_json(
            model, fan_json, fan_name=fan_name, fan_efficiency=fan_efficiency,
            pressure_rise=pressure_rise, motor_efficiency=motor_efficiency,
            motor_in_airstream_fraction=motor_in_airstream_fraction,
            end_use_subcategory=end_use_subcategory)
    elif fan_json['type'] == 'VariableVolume':
        fpc_1, fpc_2, fpc_3, fpc_4, fpc_5 = \
            _fan_curve_coefficients_from_json(fan_json['fan_curve'])
        fan = create_fan_variable_volume_from_json(
            model, fan_json, fan_name=fan_name, fan_efficiency=fan_efficiency,
            pressure_rise=pressure_rise, motor_efficiency=motor_efficiency,
            motor_in_airstream_fraction=motor_in_airstream_fraction,
            fan_power_minimum_flow_rate_input_method=fan_power_minimum_flow_rate_input_method,
            fan_power_minimum_flow_rate_fraction=fan_power_minimum_flow_rate_fraction,
            fan_power_coefficient_1=fpc_1, fan_power_coefficient_2=fpc_2,
            fan_power_coefficient_3=fpc_3, fan_power_coefficient_4=fpc_4,
            fan_power_coefficient_5=fpc_5, end_use_subcategory=end_use_subcategory)
    elif fan_json['type'] == 'ZoneExhaust':
        fan = create_fan_zone_exhaust_from_json(
            model, fan_json, fan_name=fan_name, fan_efficiency=fan_efficiency,
            pressure_rise=pressure_rise,
            system_availability_manager_coupling_mode=system_availability_manager_coupling_mode,
            end_use_subcategory=end_use_subcategory)
    return fan


def create_fan_constant_volume_from_json(
        model, fan_json, fan_name=None, fan_efficiency=None, pressure_rise=None,
        motor_efficiency=None, motor_in_airstream_fraction=None, end_use_subcategory=None):
    """Creates a constant volume fan from a json."""
    fan_efficiency = fan_json['fan_efficiency'] \
        if fan_efficiency is None else fan_efficiency
    pressure_rise = fan_json['pressure_rise'] \
        if pressure_rise is None else pressure_rise
    motor_efficiency = fan_json['motor_efficiency'] \
        if motor_efficiency is None else motor_efficiency
    motor_in_airstream_fraction = fan_json['motor_in_airstream_fraction'] \
        if motor_in_airstream_fraction is None else motor_in_airstream_fraction

    pressure_rise = PRESSURE.to_unit([pressure_rise], 'Pa', 'inH2O')[0]

    fan = openstudio_model.FanConstantVolume(model)
    _apply_base_fan_variables(
        fan, fan_name, fan_efficiency, pressure_rise, end_use_subcategory)
    if motor_efficiency is not None:
        fan.setMotorEfficiency(motor_efficiency)
    if motor_in_airstream_fraction is not None:
        fan.setMotorInAirstreamFraction(motor_in_airstream_fraction)
    return fan


def create_fan_on_off_from_json(
        model, fan_json, fan_name=None, fan_efficiency=None, pressure_rise=None,
        motor_efficiency=None, motor_in_airstream_fraction=None, end_use_subcategory=None):
    """Creates a on off fan from a json."""
    fan_efficiency = fan_json['fan_efficiency'] \
        if fan_efficiency is None else fan_efficiency
    pressure_rise = fan_json['pressure_rise'] \
        if pressure_rise is None else pressure_rise
    motor_efficiency = fan_json['motor_efficiency'] \
        if motor_efficiency is None else motor_efficiency
    motor_in_airstream_fraction = fan_json['motor_in_airstream_fraction'] \
        if motor_in_airstream_fraction is None else motor_in_airstream_fraction

    pressure_rise = PRESSURE.to_unit([pressure_rise], 'Pa', 'inH2O')[0]

    fan = openstudio_model.FanOnOff(model)
    _apply_base_fan_variables(
        fan, fan_name, fan_efficiency, pressure_rise, end_use_subcategory)
    if motor_efficiency is not None:
        fan.setMotorEfficiency(motor_efficiency)
    if motor_in_airstream_fraction is not None:
        fan.setMotorInAirstreamFraction(motor_in_airstream_fraction)
    return fan


def create_fan_variable_volume_from_json(
        model, fan_json, fan_name=None, fan_efficiency=None, pressure_rise=None,
        motor_efficiency=None, motor_in_airstream_fraction=None,
        fan_power_minimum_flow_rate_input_method=None,
        fan_power_minimum_flow_rate_fraction=None,
        end_use_subcategory=None, fan_power_coefficient_1=None,
        fan_power_coefficient_2=None, fan_power_coefficient_3=None,
        fan_power_coefficient_4=None, fan_power_coefficient_5=None):
    """Creates a variable volume fan from a json."""
    fan_efficiency = fan_json['fan_efficiency'] \
        if fan_efficiency is None else fan_efficiency
    pressure_rise = fan_json['pressure_rise'] \
        if pressure_rise is None else pressure_rise
    motor_efficiency = fan_json['motor_efficiency'] \
        if motor_efficiency is None else motor_efficiency
    motor_in_airstream_fraction = fan_json['motor_in_airstream_fraction'] \
        if motor_in_airstream_fraction is None else motor_in_airstream_fraction
    if fan_power_minimum_flow_rate_input_method is None:
        fan_power_minimum_flow_rate_input_method = \
            fan_json['fan_power_minimum_flow_rate_input_method']
    if fan_power_minimum_flow_rate_fraction is None:
        fan_power_minimum_flow_rate_fraction = \
            fan_json['fan_power_minimum_flow_rate_fraction']

    pressure_rise = PRESSURE.to_unit([pressure_rise], 'Pa', 'inH2O')[0]

    fan = openstudio_model.FanVariableVolume(model)
    _apply_base_fan_variables(
        fan, fan_name, fan_efficiency, pressure_rise, end_use_subcategory)
    if motor_efficiency is not None:
        fan.setMotorEfficiency(motor_efficiency)
    if motor_in_airstream_fraction is not None:
        fan.setMotorInAirstreamFraction(motor_in_airstream_fraction)
    if fan_power_minimum_flow_rate_input_method is not None:
        fan.setFanPowerMinimumFlowRateInputMethod(fan_power_minimum_flow_rate_input_method)
    if fan_power_minimum_flow_rate_fraction is not None:
        fan.setFanPowerMinimumFlowFraction(fan_power_minimum_flow_rate_fraction)
    if fan_power_coefficient_1 is not None:
        fan.setFanPowerCoefficient1(fan_power_coefficient_1)
    if fan_power_coefficient_2 is not None:
        fan.setFanPowerCoefficient2(fan_power_coefficient_2)
    if fan_power_coefficient_3 is not None:
        fan.setFanPowerCoefficient3(fan_power_coefficient_3)
    if fan_power_coefficient_4 is not None:
        fan.setFanPowerCoefficient4(fan_power_coefficient_4)
    if fan_power_coefficient_5 is not None:
        fan.setFanPowerCoefficient5(fan_power_coefficient_5)
    return fan


def create_fan_zone_exhaust_from_json(
        model, fan_json, fan_name=None, fan_efficiency=None, pressure_rise=None,
        system_availability_manager_coupling_mode=None, end_use_subcategory=None):
    """Creates a FanZoneExhaust from a json."""
    fan_efficiency = fan_json['fan_efficiency'] \
        if fan_efficiency is None else fan_efficiency
    pressure_rise = fan_json['pressure_rise'] \
        if pressure_rise is None else pressure_rise

    pressure_rise = PRESSURE.to_unit([pressure_rise], 'Pa', 'inH2O')[0]

    fan = openstudio_model.FanZoneExhaust(model)
    _apply_base_fan_variables(
        fan, fan_name, fan_efficiency, pressure_rise, end_use_subcategory)
    if system_availability_manager_coupling_mode is not None:
        fan.setSystemAvailabilityManagerCouplingMode(
            system_availability_manager_coupling_mode)
    return fan


def _apply_base_fan_variables(fan, fan_name=None, fan_efficiency=None,
                              pressure_rise=None, end_use_subcategory=None):
    if fan_name is not None:
        fan.setName(fan_name)
    if fan_efficiency is not None:
        fan.setFanEfficiency(fan_efficiency)
    if pressure_rise is not None:
        fan.setPressureRise(pressure_rise)
    if end_use_subcategory is not None:
        fan.setEndUseSubcategory(end_use_subcategory)
    return fan


def fan_change_impeller_efficiency(fan, impeller_eff):
    """Changes the fan impeller efficiency and also the fan total efficiency.

    Motor efficiency is preserved.

    Args:
        fan: [OpenStudio::Model::StraightComponent] Fan object. Allowable types include
            FanConstantVolume, FanOnOff, FanVariableVolume, and FanZoneExhaust.
        impeller_eff: [Double] impeller efficiency (0.0 to 1.0).
    """
    # Get the existing motor efficiency
    existing_motor_eff = 0.7
    if fan.to_FanZoneExhaust().is_initialized():
        existing_motor_eff = fan.motorEfficiency()
    # Calculate the new total efficiency
    new_total_eff = existing_motor_eff * impeller_eff
    # Set the revised motor and total fan efficiencies
    fan.setFanEfficiency(new_total_eff)


def fan_baseline_impeller_efficiency(fan):
    """Assume that the fan efficiency is 65% for normal fans and 55% for small fans."""
    fan_impeller_eff = 0.65
    if is_small_fan(fan):
        fan_impeller_eff = 0.55
    return fan_impeller_eff


def is_small_fan(fan):
    """Zone exhaust fans, FCU fans, and VAV terminal fans all count as small fans.

    Small fans get different impeller efficiencies and motor efficiencies than
    other fans.

    Args:
        fan: [OpenStudio::Model::StraightComponent] Fan object. Allowable types
            includeFanConstantVolume, FanOnOff, FanVariableVolume, and FanZoneExhaust
    """
    is_small = False
    # Exhaust fan
    if fan.to_FanZoneExhaust().is_initialized():
        is_small = True
    # Fan coil unit, unit heater, PTAC, PTHP, VRF terminals, WSHP, ERV
    elif fan.containingZoneHVACComponent().is_initialized():
        zone_hvac = fan.containingZoneHVACComponent().get()
        if zone_hvac.to_ZoneHVACFourPipeFanCoil().is_initialized():
            is_small = True
        elif zone_hvac.to_ZoneHVACPackagedTerminalAirConditioner().is_initialized() or \
                zone_hvac.to_ZoneHVACPackagedTerminalHeatPump().is_initialized() or \
                zone_hvac.to_ZoneHVACTerminalUnitVariableRefrigerantFlow().is_initialized() or \
                zone_hvac.to_ZoneHVACWaterToAirHeatPump().is_initialized() or \
                zone_hvac.to_ZoneHVACEnergyRecoveryVentilator().is_initialized():
            is_small = True
    # Powered VAV terminal
    elif fan.containingHVACComponent().is_initialized():
        zone_hvac = fan.containingHVACComponent().get()
        if zone_hvac.to_AirTerminalSingleDuctParallelPIUReheat().is_initialized() or \
                zone_hvac.to_AirTerminalSingleDuctSeriesPIUReheat().is_initialized():
            is_small = True
    return is_small
