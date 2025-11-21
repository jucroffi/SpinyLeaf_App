# coding=utf-8
"""OpenStudio load translators."""
from __future__ import division

from ladybug_geometry.geometry3d import Point3D
from honeybee.altnumber import autocalculate
from honeybee.typing import clean_ep_string
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.lib.scheduletypelimits import temperature
from honeybee_energy.lib.schedules import always_on
from honeybee_energy.load.people import People
from honeybee_energy.load.lighting import Lighting
from honeybee_energy.load.equipment import ElectricEquipment, GasEquipment
from honeybee_energy.load.process import Process
from honeybee_energy.load.hotwater import ServiceHotWater
from honeybee_energy.load.infiltration import Infiltration
from honeybee_energy.load.ventilation import Ventilation
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.load.daylight import DaylightingControl

from honeybee_openstudio.openstudio import OSPeopleDefinition, OSPeople, \
    OSLightsDefinition, OSLights, OSElectricEquipmentDefinition, OSElectricEquipment, \
    OSGasEquipmentDefinition, OSGasEquipment, OSOtherEquipmentDefinition, \
    OSOtherEquipment, OSWaterUseEquipmentDefinition, OSWaterUseEquipment, \
    OSWaterUseConnections, OSSpaceInfiltrationDesignFlowRate, \
    OSDesignSpecificationOutdoorAir, OSThermostatSetpointDualSetpoint, \
    OSZoneControlHumidistat, OSDaylightingControl, OSScheduleRuleset, os_vector_len


def _create_constant_schedule(schedule_name, schedule_value, os_model):
    """Utility function for creating a schedule with a single value."""
    # check if a constant schedule already exists and, if not, create it
    exist_schedule = os_model.getScheduleByName(schedule_name)
    if exist_schedule.is_initialized():  # return the existing schedule
        return exist_schedule.get()
    else:  # create the schedule
        os_sch = OSScheduleRuleset(os_model, schedule_value)
        os_sch.setName(schedule_name)
        return os_sch


"""____________TRANSLATORS TO OPENSTUDIO____________"""


def people_to_openstudio(people, os_model):
    """Convert Honeybee People object to OpenStudio People object."""
    # create people OpenStudio object and set identifier
    os_people_def = OSPeopleDefinition(os_model)
    os_people = OSPeople(os_people_def)
    os_people_def.setName(people.identifier)
    os_people.setName(people.identifier)
    if people._display_name is not None:
        os_people_def.setDisplayName(people.display_name)
        os_people.setDisplayName(people.display_name)
    # assign people per space floor area
    os_people_def.setPeopleperSpaceFloorArea(people.people_per_area)
    # assign occupancy schedule
    occupancy_sch = os_model.getScheduleByName(people.occupancy_schedule.identifier)
    if occupancy_sch.is_initialized():
        occupancy_sch = occupancy_sch.get()
        os_people.setNumberofPeopleSchedule(occupancy_sch)
    # assign activity schedule
    activity_sch = os_model.getScheduleByName(people.activity_schedule.identifier)
    if activity_sch.is_initialized():
        activity_sch = activity_sch.get()
        os_people.setActivityLevelSchedule(activity_sch)
    # assign radiant and latent fractions
    os_people_def.setFractionRadiant(people.radiant_fraction)
    if people.latent_fraction == autocalculate:
        os_people_def.autocalculateSensibleHeatFraction()
    else:
        os_people_def.setSensibleHeatFraction(1.0 - float(people.latent_fraction))
    return os_people


def lighting_to_openstudio(lighting, os_model):
    """Convert Honeybee Lighting object to OpenStudio Lights object."""
    # create people OpenStudio object and set identifier
    os_lighting_def = OSLightsDefinition(os_model)
    os_lighting = OSLights(os_lighting_def)
    os_lighting_def.setName(lighting.identifier)
    os_lighting.setName(lighting.identifier)
    if lighting._display_name is not None:
        os_lighting_def.setDisplayName(lighting.display_name)
        os_lighting.setDisplayName(lighting.display_name)
    # assign watts per space floor area
    os_lighting_def.setWattsperSpaceFloorArea(lighting.watts_per_area)
    # assign lighting schedule
    lighting_schedule = os_model.getScheduleByName(lighting.schedule.identifier)
    if lighting_schedule.is_initialized():
        lighting_schedule = lighting_schedule.get()
        os_lighting.setSchedule(lighting_schedule)
    # assign visible, radiant, and return air fractions
    os_lighting_def.setFractionVisible(lighting.visible_fraction)
    os_lighting_def.setFractionRadiant(lighting.radiant_fraction)
    os_lighting_def.setReturnAirFraction(lighting.return_air_fraction)
    return os_lighting


def _equipment_to_openstudio(equipment, os_equip_def, os_equip, os_model):
    """Process any type of equipment object to OpenStudio."""
    os_equip_def.setName(equipment.identifier)
    os_equip.setName(equipment.identifier)
    if equipment._display_name is not None:
        os_equip_def.setDisplayName(equipment.display_name)
        os_equip.setDisplayName(equipment.display_name)
    # assign schedule
    equip_schedule = os_model.getScheduleByName(equipment.schedule.identifier)
    if equip_schedule.is_initialized():
        equip_schedule = equip_schedule.get()
        os_equip.setSchedule(equip_schedule)
    # assign radiant, latent, and lost fractions
    os_equip_def.setFractionRadiant(equipment.radiant_fraction)
    os_equip_def.setFractionLatent(equipment.latent_fraction)
    os_equip_def.setFractionLost(equipment.lost_fraction)


def electric_equipment_to_openstudio(equipment, os_model):
    """Convert Honeybee ElectricEquipment object to OpenStudio ElectricEquipment object.
    """
    # create the OpenStudio object
    os_equip_def = OSElectricEquipmentDefinition(os_model)
    os_equip = OSElectricEquipment(os_equip_def)
    _equipment_to_openstudio(equipment, os_equip_def, os_equip, os_model)
    # assign watts per area
    os_equip_def.setWattsperSpaceFloorArea(equipment.watts_per_area)
    # ensure that it's always reported under electric equipment
    os_equip.setEndUseSubcategory('Electric Equipment')
    return os_equip


def gas_equipment_to_openstudio(equipment, os_model):
    """Convert Honeybee GasEquipment object to OpenStudio GasEquipment object.
    """
    # create the OpenStudio object
    os_equip_def = OSGasEquipmentDefinition(os_model)
    os_equip = OSGasEquipment(os_equip_def)
    _equipment_to_openstudio(equipment, os_equip_def, os_equip, os_model)
    # assign watts per area
    os_equip_def.setWattsperSpaceFloorArea(equipment.watts_per_area)
    # ensure that it's always reported under electric equipment
    os_equip.setEndUseSubcategory('Electric Equipment')
    return os_equip


def process_to_openstudio(process, os_model):
    """Convert Honeybee Process object to OpenStudio OtherEquipment object.
    """
    # create the OpenStudio object
    os_equip_def = OSOtherEquipmentDefinition(os_model)
    os_equip = OSOtherEquipment(os_equip_def)
    _equipment_to_openstudio(process, os_equip_def, os_equip, os_model)
    # assign watts
    os_equip_def.setDesignLevel(process.watts)
    # assign the fuel type and end use category
    os_equip.setFuelType(process.fuel_type)
    os_equip.setEndUseSubcategory(process.end_use_category)
    return os_equip


def hot_water_to_openstudio(hot_water, room, os_model):
    """Convert Honeybee ServiceHotWater to OpenStudio WaterUseConnections.

    Args:
        hot_water: The Honeybee-energy ServiceHotWater object to be translated
            to OpenStudio.
        room: The Honeybee Room object to which the ServiceHotWater is assigned.
            This is required in order to compute the total flow rate from the
            floor area.
        os_model: The OpenStudio Model to which the WaterUseEquipment and
            WaterUseConnections objects will be added.
    """
    # create water use equipment + connection and set identifier
    os_shw_def = OSWaterUseEquipmentDefinition(os_model)
    os_shw = OSWaterUseEquipment(os_shw_def)
    os_shw_conn = OSWaterUseConnections(os_model)
    os_shw_conn.addWaterUseEquipment(os_shw)
    u_id = '{}..{}'.format(hot_water.identifier, room.identifier)
    os_shw_def.setName(u_id)
    os_shw.setName(u_id)
    os_shw_conn.setName(u_id)
    if hot_water._display_name is not None:
        os_shw_def.setDisplayName(hot_water.display_name)
        os_shw.setDisplayName(hot_water.display_name)
        os_shw_conn.setDisplayName(hot_water.display_name)
    # assign the flow of water
    total_flow = (hot_water.flow_per_area / 3600000.) * room.floor_area
    os_shw_def.setPeakFlowRate(total_flow)
    os_shw_def.setEndUseSubcategory('General')
    # assign schedule
    shw_schedule = os_model.getScheduleByName(hot_water.schedule.identifier)
    if shw_schedule.is_initialized():
        shw_schedule = shw_schedule.get()
        os_shw.setFlowRateFractionSchedule(shw_schedule)
    # assign sensible fraction if it exists
    sens_fract = round(hot_water.sensible_fraction)
    sens_sch_name = '{} Hot Water Sensible Fraction'.format(sens_fract)
    sens_fract_sch = _create_constant_schedule(sens_sch_name, sens_fract, os_model)
    os_shw_def.setSensibleFractionSchedule(sens_fract_sch)
    # assign latent fraction if it exists
    lat_fract = round(hot_water.latent_fraction, 3)
    lat_sch_name = '{} Hot Water Latent Fraction'.format(hot_water.latent_fraction)
    lat_fract_sch = _create_constant_schedule(lat_sch_name, lat_fract, os_model)
    os_shw_def.setLatentFractionSchedule(lat_fract_sch)
    # assign the hot water temperature
    target_temp = round(hot_water.target_temperature, 3)
    target_sch_name = '{}C Hot Water'.format(target_temp)
    target_water_sch = _create_constant_schedule(target_sch_name, target_temp, os_model)
    os_shw_def.setTargetTemperatureSchedule(target_water_sch)
    # create the hot water connection with same temperature as target temperature
    os_shw_conn.setHotWaterSupplyTemperatureSchedule(target_water_sch)
    return os_shw_conn


def infiltration_to_openstudio(infiltration, os_model):
    """Convert Honeybee Infiltration to OpenStudio SpaceInfiltrationDesignFlowRate."""
    # create infiltration OpenStudio object and set identifier
    os_inf = OSSpaceInfiltrationDesignFlowRate(os_model)
    os_inf.setName(infiltration.identifier)
    if infiltration._display_name is not None:
        os_inf.setDisplayName(infiltration.display_name)
    # assign flow per surface
    os_inf.setFlowperExteriorSurfaceArea(infiltration.flow_per_exterior_area)
    # assign schedule
    inf_schedule = os_model.getScheduleByName(infiltration.schedule.identifier)
    if inf_schedule.is_initialized():
        inf_schedule = inf_schedule.get()
        os_inf.setSchedule(inf_schedule)
    # assign constant, temperature, and velocity coefficients
    os_inf.setConstantTermCoefficient(infiltration.constant_coefficient)
    os_inf.setTemperatureTermCoefficient(infiltration.temperature_coefficient)
    os_inf.setVelocityTermCoefficient(infiltration.velocity_coefficient)
    return os_inf


def ventilation_to_openstudio(ventilation, os_model):
    """Convert Honeybee Ventilation to OpenStudio DesignSpecificationOutdoorAir."""
    # create ventilation OpenStudio object and set identifier
    os_vent = OSDesignSpecificationOutdoorAir(os_model)
    os_vent.setName(ventilation.identifier)
    if ventilation._display_name is not None:
        os_vent.setDisplayName(ventilation.display_name)
    # assign air changes per hour if it exists
    os_vent.setOutdoorAirFlowAirChangesperHour(ventilation.air_changes_per_hour)
    os_vent.setOutdoorAirFlowRate(ventilation.flow_per_zone)
    os_vent.setOutdoorAirFlowperPerson(ventilation.flow_per_person)
    os_vent.setOutdoorAirFlowperFloorArea(ventilation.flow_per_area)
    # set the schedule if it exists
    if ventilation.schedule is not None:
        vent_sch = os_model.getScheduleByName(ventilation.schedule.identifier)
        if vent_sch.is_initialized():
            vent_sch = vent_sch.get()
            os_vent.setOutdoorAirFlowRateFractionSchedule(vent_sch)
    return os_vent


def setpoint_to_openstudio_thermostat(setpoint, os_model, zone_identifier=None):
    """Convert Honeybee Setpoint to OpenStudio ThermostatSetpointDualSetpoint."""
    # create thermostat OpenStudio object and set identifier
    os_thermostat = OSThermostatSetpointDualSetpoint(os_model)
    if zone_identifier:
        os_thermostat.setName('{}..{}'.format(setpoint.identifier, zone_identifier))
    else:
        os_thermostat.setName(setpoint.identifier)
    if setpoint._display_name is not None:
        os_thermostat.setDisplayName(setpoint.display_name)
    # assign heating schedule
    heat_schedule = os_model.getScheduleByName(setpoint.heating_schedule.identifier)
    if heat_schedule.is_initialized():
        heat_schedule = heat_schedule.get()
        os_thermostat.setHeatingSetpointTemperatureSchedule(heat_schedule)
    # assign cooling schedule
    cool_schedule = os_model.getScheduleByName(setpoint.cooling_schedule.identifier)
    if cool_schedule.is_initialized():
        cool_schedule = cool_schedule.get()
        os_thermostat.setCoolingSetpointTemperatureSchedule(cool_schedule)
    # assign the setpoint_cutout_difference
    if setpoint.setpoint_cutout_difference != 0:
        os_thermostat.setTemperatureDifferenceBetweenCutoutAndSetpoint(
            setpoint.setpoint_cutout_difference)
    return os_thermostat


def setpoint_to_openstudio_humidistat(setpoint, os_model, zone_identifier=None):
    """Convert Honeybee Setpoint to OpenStudio ZoneControlHumidistat."""
    # create a humidistat if specified
    if setpoint.humidifying_schedule is not None:
        os_humidistat = OSZoneControlHumidistat(os_model)
        if zone_identifier:
            os_humidistat.setName('{}..{}'.format(setpoint.identifier, zone_identifier))
        else:
            os_humidistat.setName(setpoint.identifier)
        if setpoint._display_name is not None:
            os_humidistat.setDisplayName(setpoint.display_name)
        # assign heating schedule
        humid_sch = os_model.getScheduleByName(setpoint.humidifying_schedule.identifier)
        if humid_sch.is_initialized():
            humid_sch = humid_sch.get()
            os_humidistat.setHumidifyingRelativeHumiditySetpointSchedule(humid_sch)
        # assign cooling schedule
        dehumid_sch = os_model.getScheduleByName(setpoint.dehumidifying_schedule.identifier)
        if dehumid_sch.is_initialized():
            dehumid_sch = dehumid_sch.get()
            os_humidistat.setDehumidifyingRelativeHumiditySetpointSchedule(dehumid_sch)
        return os_humidistat


def daylight_to_openstudio(daylight, os_model):
    """Convert Honeybee DaylightingControl to OpenStudio DaylightingControl."""
    # create daylighting control OpenStudio object and set identifier
    os_daylight = OSDaylightingControl(os_model)
    # assign the position of the sensor point
    os_daylight.setPositionXCoordinate(daylight.sensor_position.x)
    os_daylight.setPositionYCoordinate(daylight.sensor_position.y)
    os_daylight.setPositionZCoordinate(daylight.sensor_position.z)
    # set all of the other properties of the daylight control
    os_daylight.setIlluminanceSetpoint(daylight.illuminance_setpoint)
    os_daylight.setMinimumInputPowerFractionforContinuousDimmingControl(
        daylight.min_power_input)
    os_daylight.setMinimumLightOutputFractionforContinuousDimmingControl(
        daylight.min_light_output)
    if daylight.off_at_minimum:
        os_daylight.setLightingControlType('Continuous/Off')
    else:
        os_daylight.setLightingControlType('Continuous')
    return os_daylight


"""____________TRANSLATORS FROM OPENSTUDIO____________"""


def people_from_openstudio(os_people, schedules=None):
    """Convert OpenStudio People object to Honeybee People."""
    # create the people object
    load_def = os_people.peopleDefinition()
    people_per_area = float(str(load_def.peopleperSpaceFloorArea())) \
        if load_def.peopleperSpaceFloorArea().is_initialized() else 0
    occupancy_schedule = always_on
    if schedules is not None and os_people.numberofPeopleSchedule().is_initialized():
        schedule = os_people.numberofPeopleSchedule().get()
        try:
            occupancy_schedule = schedules[schedule.nameString()]
        except KeyError:
            occupancy_schedule = always_on
    people = People(clean_ep_string(os_people.nameString()),
                    people_per_area, occupancy_schedule)
    # assign the optional attributes
    if os_people.activityLevelSchedule().is_initialized():
        schedule = os_people.activityLevelSchedule().get()
        try:
            people.activity_schedule = schedules[schedule.nameString()]
        except KeyError:
            pass
    people.radiant_fraction = load_def.fractionRadiant()
    if not load_def.isSensibleHeatFractionAutocalculated():
        sensible_fraction = load_def.sensibleHeatFraction()
        if sensible_fraction.is_initialized():
            people.latent_fraction = 1 - sensible_fraction.get()
    if os_people.displayName().is_initialized():
        people.display_name = os_people.displayName().get()
    return people


def lighting_from_openstudio(os_lighting, schedules=None):
    """Convert OpenStudio Lights object to Honeybee Lighting."""
    # create the lighting object
    loads_def = os_lighting.lightsDefinition()
    watts_per_area = float(loads_def.wattsperSpaceFloorArea().get()) \
        if loads_def.wattsperSpaceFloorArea().is_initialized() else 0
    schedule = always_on
    if schedules is not None and os_lighting.schedule().is_initialized():
        schedule = os_lighting.schedule().get()
        try:
            schedule = schedules[schedule.nameString()]
        except KeyError:
            schedule = always_on
    lighting = Lighting(clean_ep_string(os_lighting.nameString()),
                        watts_per_area, schedule)
    # assign the optional attributes
    lighting.visible_fraction = float(loads_def.fractionVisible())
    lighting.radiant_fraction = float(loads_def.fractionRadiant())
    lighting.return_air_fraction = float(loads_def.returnAirFraction())
    if os_lighting.displayName().is_initialized():
        lighting.display_name = os_lighting.displayName().get()
    return lighting


def electric_equipment_from_openstudio(os_equipment, schedules=None):
    """Convert OpenStudio ElectricEquipment object to Honeybee ElectricEquipment."""
    # create the equipment object
    loads_def = os_equipment.electricEquipmentDefinition()
    watts_per_area = float(loads_def.wattsperSpaceFloorArea().get()) \
        if loads_def.wattsperSpaceFloorArea().is_initialized() else 0
    schedule = always_on
    if schedules is not None and os_equipment.schedule().is_initialized():
        schedule = os_equipment.schedule().get()
        try:
            schedule = schedules[schedule.nameString()]
        except KeyError:
            schedule = always_on
    equipment = ElectricEquipment(clean_ep_string(os_equipment.nameString()),
                                  watts_per_area, schedule)
    # assign the optional attributes
    equipment.radiant_fraction = float(loads_def.fractionRadiant())
    equipment.latent_fraction = float(loads_def.fractionLatent())
    equipment.lost_fraction = float(loads_def.fractionLost())
    if os_equipment.displayName().is_initialized():
        equipment.display_name = os_equipment.displayName().get()
    return equipment


def gas_equipment_from_openstudio(os_equipment, schedules=None):
    """Convert OpenStudio GasEquipment object to Honeybee GasEquipment."""
    # create the equipment object
    loads_def = os_equipment.gasEquipmentDefinition()
    watts_per_area = float(loads_def.wattsperSpaceFloorArea().get()) \
        if loads_def.wattsperSpaceFloorArea().is_initialized() else 0
    schedule = always_on
    if schedules is not None and os_equipment.schedule().is_initialized():
        schedule = os_equipment.schedule().get()
        try:
            schedule = schedules[schedule.nameString()]
        except KeyError:
            schedule = always_on
    equipment = GasEquipment(clean_ep_string(os_equipment.nameString()),
                             watts_per_area, schedule)
    # assign the optional attributes
    equipment.radiant_fraction = float(loads_def.fractionRadiant())
    equipment.latent_fraction = float(loads_def.fractionLatent())
    equipment.lost_fraction = float(loads_def.fractionLost())
    if os_equipment.displayName().is_initialized():
        equipment.display_name = os_equipment.displayName().get()
    return equipment


def process_from_openstudio(os_process, schedules=None):
    """Convert OpenStudio OtherEquipment object to Honeybee Process."""
    # create the process object
    loads_def = os_process.otherEquipmentDefinition()
    watts = float(loads_def.designLevel().get()) \
        if loads_def.designLevel().is_initialized() else 0
    schedule = always_on
    if schedules is not None and os_process.schedule().is_initialized():
        schedule = os_process.schedule().get()
        try:
            schedule = schedules[schedule.nameString()]
        except KeyError:
            schedule = always_on
    fuel_type = os_process.fuelType()
    equipment = Process(clean_ep_string(os_process.nameString()),
                        watts, schedule, fuel_type)
    # assign the optional attributes
    equipment.end_use_category = os_process.endUseSubcategory()
    equipment.radiant_fraction = float(loads_def.fractionRadiant())
    equipment.latent_fraction = float(loads_def.fractionLatent())
    equipment.lost_fraction = float(loads_def.fractionLost())
    if os_process.displayName().is_initialized():
        equipment.display_name = os_process.displayName().get()
    return equipment


def hot_water_from_openstudio(os_hot_water, floor_area, schedules=None):
    """Convert list of OpenStudio WaterUseEquipment to Honeybee ServiceHotWater."""
    if isinstance(os_hot_water, (list, tuple)) and len(os_hot_water) == 0:
        return None
    elif os_vector_len(os_hot_water) == 0:
        return None
    # sum up all of the flow rates
    flow_per_area, schedule = 0, always_on
    for load in os_hot_water:
        load_def = load.waterUseEquipmentDefinition()
        peak_flow = load_def.peakFlowRate()
        # unit for flow per area is L/h-m2 (m3/s = 3600000 L/h)
        flow_per_area += ((peak_flow * 3600000.) / floor_area)
        if schedules is not None and load.flowRateFractionSchedule().is_initialized():
            schedule = load.flowRateFractionSchedule().get()
            try:
                schedule = schedules[schedule.nameString()]
            except KeyError:
                pass
    hot_water = ServiceHotWater(clean_ep_string(load.nameString()),
                                flow_per_area, schedule)
    # assign the optional attributes
    if load.displayName().is_initialized():
        hot_water.display_name = load.displayName().get()
    return hot_water


def infiltration_from_openstudio(os_infiltration, schedules=None):
    """Convert OpenStudio SpaceInfiltrationDesignFlowRate to Honeybee Infiltration."""
    # create the infiltration object
    flow_per_exterior_area = os_infiltration.flowperExteriorSurfaceArea().get() \
        if os_infiltration.flowperExteriorSurfaceArea().is_initialized() else 0
    schedule = always_on
    if schedules is not None and os_infiltration.schedule().is_initialized():
        try:
            schedule = schedules[os_infiltration.schedule().get().nameString()]
        except KeyError:
            schedule = always_on
    infiltration = Infiltration(clean_ep_string(os_infiltration.nameString()),
                                flow_per_exterior_area, schedule)
    # assign the optional attributes
    infiltration.constant_coefficient = os_infiltration.constantTermCoefficient()
    infiltration.temperature_coefficient = os_infiltration.temperatureTermCoefficient()
    infiltration.velocity_coefficient = os_infiltration.velocityTermCoefficient()
    if os_infiltration.displayName().is_initialized():
        infiltration.display_name = os_infiltration.displayName().get()
    return infiltration


def ventilation_from_openstudio(os_ventilation, schedules=None):
    """Convert OpenStudio DesignSpecificationOutdoorAir to Honeybee Ventilation."""
    # create the ventilation object
    ventilation = Ventilation(clean_ep_string(os_ventilation.nameString()))
    # assign the optional attributes
    ventilation.air_changes_per_hour = os_ventilation.outdoorAirFlowAirChangesperHour()
    ventilation.flow_per_zone = os_ventilation.outdoorAirFlowRate()
    ventilation.flow_per_person = os_ventilation.outdoorAirFlowperPerson()
    ventilation.flow_per_area = os_ventilation.outdoorAirFlowperFloorArea()
    if schedules is not None and \
            os_ventilation.outdoorAirFlowRateFractionSchedule().is_initialized():
        schedule = os_ventilation.outdoorAirFlowRateFractionSchedule().get()
        try:
            ventilation.schedule = schedules[schedule.nameString()]
        except KeyError:
            pass
    if os_ventilation.displayName().is_initialized():
        ventilation.display_name = os_ventilation.displayName().get()
    return ventilation


def setpoint_from_openstudio_thermostat(os_thermostat, schedules=None):
    """Convert OpenStudio ThermostatSetpointDualSetpoint to Honeybee Setpoint."""
    # create the setpoint object
    heat_sch = None
    if os_thermostat.heatingSetpointTemperatureSchedule().is_initialized():
        heat_sch = os_thermostat.heatingSetpointTemperatureSchedule().get()
        try:
            heat_sch = schedules[heat_sch.nameString()]
        except KeyError:
            heat_sch = None
    if heat_sch is None:
        heat_sch = ScheduleRuleset.from_constant_value('No Heating', -100, temperature)
    cool_sch = None
    if os_thermostat.coolingSetpointTemperatureSchedule().is_initialized():
        cool_sch = os_thermostat.coolingSetpointTemperatureSchedule().get()
        try:
            cool_sch = schedules[cool_sch.nameString()]
        except KeyError:
            cool_sch = None
    if cool_sch is None:
        cool_sch = ScheduleRuleset.from_constant_value('No Cooling', 100, temperature)
    setpoint = Setpoint(clean_ep_string(os_thermostat.nameString()), heat_sch, cool_sch)
    # assign the optional attributes
    if os_thermostat.displayName().is_initialized():
        setpoint.display_name = os_thermostat.displayName().get()
    return setpoint


def setpoint_from_openstudio_humidistat(os_humidistat, setpoint, schedules=None):
    """Convert OpenStudio ZoneControlHumidistat to Honeybee Setpoint."""
    # create the setpoint object
    if os_humidistat.humidifyingRelativeHumiditySetpointSchedule().is_initialized():
        humid_sch = os_humidistat.humidifyingRelativeHumiditySetpointSchedule().get()
        try:
            setpoint.humidifying_schedule = schedules[humid_sch.nameString()]
        except KeyError:
            pass
    if os_humidistat.dehumidifyingRelativeHumiditySetpointSchedule().is_initialized():
        dehumid_sch = os_humidistat.dehumidifyingRelativeHumiditySetpointSchedule().get()
        try:
            setpoint.dehumidifying_schedule = schedules[dehumid_sch.nameString()]
        except KeyError:
            pass
    return setpoint


def daylight_from_openstudio(os_daylight):
    """Convert OpenStudio DaylightingControl to Honeybee DaylightingControl."""
    # create the daylight control object
    sensor_position = Point3D(
        os_daylight.positionXCoordinate(), os_daylight.positionYCoordinate(),
        os_daylight.positionZCoordinate())
    daylight = DaylightingControl(sensor_position)
    # assign all of the optional attributes
    daylight.illuminance_setpoint = os_daylight.illuminanceSetpoint()
    daylight.min_power_input = \
        os_daylight.minimumInputPowerFractionforContinuousDimmingControl()
    daylight.min_light_output = \
        os_daylight.minimumLightOutputFractionforContinuousDimmingControl()
    control_type = os_daylight.lightingControlType().lower()
    daylight.off_at_minimum = True if control_type == 'continuous/off' else False
    return daylight
