# coding=utf-8
"""OpenStudio ventilative cooling translators."""
from __future__ import division
import re

from honeybee.boundarycondition import Outdoors
from honeybee_energy.ventcool.control import VentilationControl

from honeybee_openstudio.openstudio import OSZoneVentilationWindandStackOpenArea, \
    OSZoneVentilationDesignFlowRate, OSAirflowNetworkCrack, OSAirflowNetworkSimpleOpening, \
    OSAirflowNetworkHorizontalOpening, OSAirflowNetworkReferenceCrackConditions, \
    OSOutputVariable, OSEnergyManagementSystemSensor, OSEnergyManagementSystemActuator, \
    OSEnergyManagementSystemProgram, OSEnergyManagementSystemProgramCallingManager


def ventilation_opening_to_openstudio(opening, os_model):
    """Convert VentilationOpening to OpenStudio ZoneVentilationWindandStackOpenArea.

    Args:
        opening: The Honeybee VentilationOpening object to be translated
            to OpenStudio. Note that this object must be assigned to a parent
            Aperture with a parent Room in order to be successfully translated.
        os_model: The OpenStudio model to which the ZoneVentilationWindandStackOpenArea
            will be added.
    """
    # check that a parent is assigned
    assert opening.parent is not None, 'VentilationOpening must be assigned ' \
        'to an Aperture or Door to translate to_openstudio.'
    # get the VentilationControl object from the room
    control = None
    room = None
    if opening.parent.has_parent:
        if opening.parent.parent.has_parent:
            room = opening.parent.parent.parent
            if room.properties.energy.window_vent_control is not None:
                control = room.properties.energy.window_vent_control
    if control is None:  # use default ventilation control
        control = VentilationControl()
    assert room is not None, 'VentilationOpening must have a parent Room to ' \
        'translate to_openstudio.'
    # process the properties on this object into IDF format
    angle = opening.parent.horizontal_orientation() \
        if opening.parent.normal.z != 1 else 0
    angle = angle % 360
    height = (opening.parent.geometry.max.z - opening.parent.geometry.min.z) * \
        opening.fraction_height_operable
    # create wind and stack object and set all of its properties
    os_opening = OSZoneVentilationWindandStackOpenArea(os_model)
    os_opening.setName('{}_Opening'.format(opening.parent.identifier))
    os_opening.setOpeningArea(opening.parent.area * opening.fraction_area_operable)
    os_opening.setHeightDifference(height)
    os_opening.setEffectiveAngle(angle)
    os_opening.setDischargeCoefficientforOpening(opening.discharge_coefficient)
    if opening.wind_cross_vent:
        os_opening.autocalculateOpeningEffectiveness()
    else:
        os_opening.setOpeningEffectiveness(0)
    # set the properties of the ventilation control
    os_opening.setMinimumIndoorTemperature(control.min_indoor_temperature)
    os_opening.setMaximumIndoorTemperature(control.max_indoor_temperature)
    os_opening.setMinimumOutdoorTemperature(control.min_outdoor_temperature)
    os_opening.setMaximumOutdoorTemperature(control.max_outdoor_temperature)
    os_opening.setDeltaTemperature(control.delta_temperature)
    if control.schedule.identifier != 'Always On':
        vent_sch = os_model.getScheduleByName(control.schedule.identifier)
        if vent_sch.is_initialized():
            os_vent_sch = vent_sch.get()
            os_opening.setOpeningAreaFractionSchedule(os_vent_sch)
    return os_opening


def ventilation_fan_to_openstudio(fan, os_model):
    """Convert VentilationFan to OpenStudio ZoneVentilationDesignFlowRate."""
    # create zone ventilation object and set identifier
    os_fan = OSZoneVentilationDesignFlowRate(os_model)
    os_fan.setName(fan.identifier)
    if fan._display_name is not None:
        os_fan.setDisplayName(fan.display_name)
    # assign fan properties
    os_fan.setDesignFlowRate(fan.flow_rate)
    os_fan.setFanPressureRise(fan.pressure_rise)
    os_fan.setFanTotalEfficiency(fan.efficiency)
    os_fan.setVentilationType(fan.ventilation_type)
    # set all of the ventilation control properties
    os_fan.setMinimumIndoorTemperature(fan.control.min_indoor_temperature)
    os_fan.setMaximumIndoorTemperature(fan.control.max_indoor_temperature)
    os_fan.setMinimumOutdoorTemperature(fan.control.min_outdoor_temperature)
    os_fan.setMaximumOutdoorTemperature(fan.control.max_outdoor_temperature)
    os_fan.setDeltaTemperature(fan.control.delta_temperature)
    # assign schedule if it exists
    if fan.control.schedule.identifier != 'Always On':
        vent_sch = os_model.getScheduleByName(fan.control.schedule.identifier)
        if vent_sch.is_initialized():
            os_vent_sch = vent_sch.get()
            os_fan.setSchedule(os_vent_sch)
    return os_fan


def ventilation_sim_control_to_openstudio(vent_sim_control, os_model):
    """Convert VentilationSimulationControl to OpenStudio.

    This method returns an AirflowNetworkReferenceCrackConditions that can
    be used for the rest of the AFN setup.
    """
    # create the AirflowNetworkSimulationControl object
    os_v_sim_ctrl = os_model.getAirflowNetworkSimulationControl()
    os_v_sim_ctrl.setName('Window Based Ventilative Cooling')
    os_v_sim_ctrl.setAirflowNetworkControl(vent_sim_control.vent_control_type)
    os_v_sim_ctrl.setBuildingType(vent_sim_control.building_type)
    os_v_sim_ctrl.setAzimuthAngleofLongAxisofBuilding(vent_sim_control.long_axis_angle)
    os_v_sim_ctrl.setBuildingAspectRatio(vent_sim_control.aspect_ratio)
    # create the AirflowNetworkReferenceCrackConditions that other cracks reference
    os_ref_crack = OSAirflowNetworkReferenceCrackConditions(os_model)
    os_ref_crack.setName('Reference Crack Conditions')
    os_ref_crack.setTemperature(vent_sim_control.reference_temperature)
    os_ref_crack.setBarometricPressure(vent_sim_control.reference_pressure)
    os_ref_crack.setHumidityRatio(vent_sim_control.reference_humidity_ratio)
    return os_ref_crack


def afn_crack_to_openstudio(afn_crack, os_model, os_reference_crack=None):
    """Convert Honeybee AFNCrack to OpenStudio AirflowNetworkCrack.

    Args:
        opening: The Honeybee VentilationOpening object to be translated
            to OpenStudio. Note that this object must be assigned to a parent
            Aperture with a parent Room in order to be successfully translated.
        os_model: The OpenStudio model to which the AirflowNetworkSurface
            will be added.
        os_reference_crack: An optional AirflowNetworkReferenceCrackConditions
            object to set the reference. If None, a default reference crack will
            be created. (Default: None).
    """
    flow_coefficient = afn_crack.flow_coefficient \
        if afn_crack.flow_coefficient > 1.0e-09 else 1.0e-09
    flow_exponent = afn_crack.flow_exponent
    if os_reference_crack is None:
        os_reference_crack = _default_reference_crack(os_model)
    os_crack = OSAirflowNetworkCrack(
        os_model, flow_coefficient, flow_exponent, os_reference_crack)
    return os_crack


def ventilation_opening_to_openstudio_afn(opening, os_model, os_reference_crack=None):
    """Convert Honeybee VentilationOpening to OpenStudio AirflowNetworkSimpleOpening.

    The returned output may also be a AirflowNetworkHorizontalOpening or a
    AirflowNetworkCrack if the opening is assigned to a parent Aperture or Door
    that is horizontal and so it cannot be represented with AirflowNetworkSimpleOpening.

    Args:
        opening: The Honeybee VentilationOpening object to be translated
            to OpenStudio. Note that this object must be assigned to a parent
            Aperture with a parent Room in order to be successfully translated.
        os_model: The OpenStudio model to which the AirflowNetworkSimpleOpening
            will be added.
        os_reference_crack: An optional AirflowNetworkReferenceCrackConditions
            object to set the reference when the ventilation opening is being
            translated to a large crack. This happens when the ventilation
            opening is horizontal and in an outdoor Face. If None, a default
            reference crack will be created. (Default: None).

    Returns:
        A tuple with two elements.

        -   os_opening -- The OpenStudio AFN SimpleOpening, HorizontalOpening
            or Crack that represents the ventilation opening.

        -   opening_factor - A number for the opening factor to be assigned to
            the parent OpenStudio AirflowNetworkSurface and incorporated into
            the EMS program.
    """
    # check that a parent is assigned
    assert opening.parent is not None, 'VentilationOpening must be assigned ' \
        'to an Aperture or Door to translate to_openstudio.'
    # get the tilt and BC of the parent so that we can use the correct AFN object
    srf_tilt = opening.parent.tilt
    srf_bc = opening.parent.boundary_condition
    # process the flow coefficient, flow exponent and fraction area operable
    flow_coeff = opening.flow_coefficient_closed \
        if opening.flow_coefficient_closed > 1.0e-09 else 1.0e-09
    flow_exponent = opening.flow_exponent_closed
    discharge_coeff = opening.discharge_coefficient
    two_way_thresh = opening.two_way_threshold
    opening_factor = opening.fraction_area_operable
    # create an opening obj
    if srf_tilt < 10 or srf_tilt > 170:
        if isinstance(srf_bc, Outdoors):
            # create a crack to represent an exterior in-operable horizontal skylight
            opening_factor = None
            if os_reference_crack is None:
                os_reference_crack = _default_reference_crack(os_model)
            os_opening = OSAirflowNetworkCrack(
                os_model, flow_coeff, flow_exponent, os_reference_crack)
        else:
            # create a HorizontalOpening object to for the interior horizontal window
            slope_ang = 90 - srf_tilt if srf_tilt < 10 else 90 - (180 - srf_tilt)
            os_opening = OSAirflowNetworkHorizontalOpening(
                os_model, flow_coeff, flow_exponent, slope_ang, discharge_coeff)
    else:
        # create the simple opening object for the Aperture or Door using default values
        os_opening = OSAirflowNetworkSimpleOpening(
          os_model, flow_coeff, flow_exponent, two_way_thresh, discharge_coeff)
    os_opening.setName('{}_Opening'.format(opening.parent.identifier))
    return os_opening, opening_factor


def ventilation_control_to_openstudio_afn(
        control, open_factors, os_sub_faces, os_zone_air_temp, os_model, room_id=''):
    """Convert Honeybee VentilationControl to OpenStudio EnergyManagementSystemProgram.

    Args:
        control: The Honeybee VentilationControl object to be translated to OpenStudio.
        open_factors: A list of numbers for the opening factor of each Subface
            to be controlled by the VentilationControl.
        os_sub_faces: A list of OpenStudio SubSurface objects that have AFN
            SimpleOpening or HorizontalOpening objects to be controlled by the EMS.
        os_zone_air_temp: The OpenStudio EnergyManagementSystemSensor object for the
            Zone Air Temperature that corresponds with the VentilationControl.
            If this sensor does not yet exist in the model, the zone_temperature_sensor
            function in this module can be used to create it.
        os_model: The OpenStudio model to which the EnergyManagementSystemProgram
            will be added.
        room_id: An optional Room identifier to be used to ensure the names used
            in the resulting EnergyManagementSystemProgram are unique to the
            Room to which the VentilationControl is applied.
    """
    # set up a schedule sensor if there's a schedule specified
    sch_sen_id = None
    if control.schedule.identifier != 'Always On':
        vent_sch = os_model.getScheduleByName(control.schedule.identifier)
        if vent_sch.is_initialized():
            sch_var = OSOutputVariable('Schedule Value', os_model)
            sch_var.setReportingFrequency('Timestep')
            sch_var.setKeyValue(control.schedule.identifier)
            sch_sensor = OSEnergyManagementSystemSensor(os_model, sch_var)
            sch_sen_id = 'SensorSch{}'.format(re.sub('[^A-Za-z0-9]', '', room_id))
            sch_sensor.setName(sch_sen_id)

    # create the actuators for each of the operable windows
    actuator_ids = []
    for os_sub_f in os_sub_faces:
        window_actuator = OSEnergyManagementSystemActuator(
            os_sub_f, 'AirFlow Network Window/Door Opening',
            'Venting Opening Factor')
        act_id = 'OpenFactor{}'.format(re.sub('[^A-Za-z0-9]', '', os_sub_f.nameString()))
        window_actuator.setName(act_id)
        actuator_ids.append(act_id)

    # create the first part of the EMS Program to open windows according to control logic
    logic_statements = []
    in_sen_id = os_zone_air_temp.nameString()
    min_in = control.min_indoor_temperature
    max_in = control.max_indoor_temperature
    min_out = control.min_outdoor_temperature
    max_out = control.max_outdoor_temperature
    d_in_out = control.delta_temperature
    if min_in != -100:
        logic_statements.append('({} > {})'.format(in_sen_id, min_in))
    if max_in != 100:
        logic_statements.append('({} < {})'.format(in_sen_id, max_in))
    if min_out != -100:
        logic_statements.append('(Outdoor_Sensor > {})'.format(min_out))
    if max_out != 100:
        logic_statements.append('(Outdoor_Sensor < {})'.format(max_out))
    if d_in_out != -100:
        logic_statements.append('(({} - Outdoor_Sensor) > {})'.format(in_sen_id, d_in_out))
    if sch_sen_id is not None:
        logic_statements.append('({} > 0)'.format(sch_sen_id))
    if len(logic_statements) == 0:  # no logic has been provided; always open windows
        complete_logic = 'IF (Outdoor_Sensor < 100)'
    else:
        complete_logic = 'IF {}'.format(' && '.join(logic_statements))

    # initialize the program and add the complete logic
    ems_program = OSEnergyManagementSystemProgram(os_model)
    prog_name = 'WindowOpening{}'.format(re.sub('[^A-Za-z0-9]', '', room_id))
    ems_program.setName(prog_name)
    ems_program.addLine(complete_logic)

    # loop through each of the actuators and open/close each window
    for act_id, open_factor in zip(actuator_ids, open_factors):
        ems_program.addLine('SET {} = {}'.format(act_id, open_factor))
    ems_program.addLine('ELSE')
    for act_id in actuator_ids:
        ems_program.addLine('SET {} = 0'.format(act_id))
    ems_program.addLine('ENDIF')
    return ems_program


def zone_temperature_sensor(os_zone, os_model):
    """Create an EnergyManagementSystemSensor for Zone Air Temperature."""
    os_zone_name = os_zone.nameString()
    in_var = OSOutputVariable('Zone Air Temperature', os_model)
    in_var.setReportingFrequency('Timestep')
    in_var.setKeyValue(os_zone_name)
    in_air_sensor = OSEnergyManagementSystemSensor(os_model, in_var)
    sensor_id = 'Sensor{}'.format(re.sub('[^A-Za-z0-9]', '', os_zone_name))
    in_air_sensor.setName(sensor_id)
    return in_air_sensor


def outdoor_temperature_sensor(os_model):
    """Create an EnergyManagementSystemSensor for Site Outdoor Air DryBulb Temperature.
    """
    out_var = OSOutputVariable('Site Outdoor Air Drybulb Temperature', os_model)
    out_var.setReportingFrequency('Timestep')
    out_var.setKeyValue('Environment')
    out_air_sensor = OSEnergyManagementSystemSensor(os_model, out_var)
    out_air_sensor.setName('Outdoor_Sensor')
    return out_air_sensor


def ventilation_control_program_manager(os_model):
    """Create an EMS Program Manager for all window opening."""
    os_prog_manager = OSEnergyManagementSystemProgramCallingManager(os_model)
    os_prog_manager.setName('Temperature_Controlled_Window_Opening')
    os_prog_manager.setCallingPoint('BeginTimestepBeforePredictor')
    return os_prog_manager


def _default_reference_crack(os_model):
    """Create an AFN Reference Crack with default characteristics."""
    default_name = 'Reference Crack Conditions'
    os_ref_crack = os_model.getAirflowNetworkReferenceCrackConditionsByName(default_name)
    if os_ref_crack.is_initialized():
        return os_ref_crack.get()
    os_ref_crack = OSAirflowNetworkReferenceCrackConditions(os_model)
    os_ref_crack.setTemperature(20)
    os_ref_crack.setBarometricPressure(101325)
    os_ref_crack.setHumidityRatio(0)
    return os_ref_crack
