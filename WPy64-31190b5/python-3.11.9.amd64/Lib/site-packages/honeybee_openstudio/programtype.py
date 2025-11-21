# coding=utf-8
"""OpenStudio ProgramType translator."""
from __future__ import division

from honeybee.typing import clean_ep_string
from honeybee_energy.programtype import ProgramType

from honeybee_openstudio.load import people_to_openstudio, lighting_to_openstudio, \
    electric_equipment_to_openstudio, gas_equipment_to_openstudio, \
    infiltration_to_openstudio, ventilation_to_openstudio, people_from_openstudio, \
    lighting_from_openstudio, electric_equipment_from_openstudio, \
    gas_equipment_from_openstudio, infiltration_from_openstudio, \
    ventilation_from_openstudio
from honeybee_openstudio.openstudio import OSSpaceType


def program_type_to_openstudio(program_type, os_model, include_infiltration=True):
    """Convert Honeybee ProgramType to OpenStudio SpaceType.

    Args:
        program_type: A Honeybee-energy ProgramType to be translated to OpenStudio.
        os_model: The OpenStudio Model object to which the SpaceType will be added.
        include_infiltration: Boolean for whether or not infiltration will be included
            in the translation of the ProgramType. It may be desirable to set this
            to False if the building airflow is being modeled with the EnergyPlus
            AirFlowNetwork. (Default: True).
    """
    # create openstudio space type object
    os_space_type = OSSpaceType(os_model)
    os_space_type.setName(program_type.identifier)
    if program_type._display_name is not None:
        os_space_type.setDisplayName(program_type.display_name)
    # if the program is from honeybee-energy-standards, also set the measure tag
    std_spc_type = program_type.identifier.split('::')
    if len(std_spc_type) == 3:  # originated from honeybee-energy-standards
        std_spc_type = std_spc_type[2]
        std_spc_type = std_spc_type.split('_')[0]
        os_space_type.setStandardsSpaceType(std_spc_type)
    # assign people
    if program_type.people is not None:
        os_people = people_to_openstudio(program_type.people, os_model)
        os_people.setSpaceType(os_space_type)
    # assign lighting
    if program_type.lighting is not None:
        os_lights = lighting_to_openstudio(program_type.lighting, os_model)
        os_lights.setSpaceType(os_space_type)
    # assign electric equipment
    if program_type.electric_equipment is not None:
        os_equip = electric_equipment_to_openstudio(program_type.electric_equipment, os_model)
        os_equip.setSpaceType(os_space_type)
    # assign gas equipment
    if program_type.gas_equipment is not None:
        os_equip = gas_equipment_to_openstudio(program_type.gas_equipment, os_model)
        os_equip.setSpaceType(os_space_type)
    # assign infiltration
    if program_type.infiltration is not None and include_infiltration:
        os_inf = infiltration_to_openstudio(program_type.infiltration, os_model)
        os_inf.setSpaceType(os_space_type)
    # assign ventilation
    if program_type.ventilation is not None:
        os_vent = ventilation_to_openstudio(program_type.ventilation, os_model)
        os_space_type.setDesignSpecificationOutdoorAir(os_vent)
    return os_space_type


def program_type_from_openstudio(os_space_type, schedules=None):
    """Convert OpenStudio SpaceType to Honeybee ProgramType."""
    program_type = ProgramType(clean_ep_string(os_space_type.nameString()))
    # assign people
    for os_people in os_space_type.people():
        people_def = os_people.peopleDefinition()  # only translate if people per floor
        if people_def.peopleperSpaceFloorArea().is_initialized():
            program_type.people = people_from_openstudio(os_people, schedules)
    # assign lighting
    for os_lights in os_space_type.lights():
        light_def = os_lights.lightsDefinition()  # only translate if watts per floor
        if light_def.wattsperSpaceFloorArea().is_initialized():
            program_type.lighting = lighting_from_openstudio(os_lights, schedules)
    # assign electric equipment
    for os_equip in os_space_type.electricEquipment():
        electric_eq_def = os_equip.electricEquipmentDefinition()
        if electric_eq_def.wattsperSpaceFloorArea().is_initialized():
            program_type.electric_equipment = \
                electric_equipment_from_openstudio(os_equip, schedules)
    # assign gas equipment
    for os_equip in os_space_type.gasEquipment():
        electric_eq_def = os_equip.gasEquipmentDefinition()
        if electric_eq_def.wattsperSpaceFloorArea().is_initialized():
            program_type.gas_equipment = \
                gas_equipment_from_openstudio(os_equip, schedules)
    # assign infiltration
    for os_inf in os_space_type.spaceInfiltrationDesignFlowRates():
        if os_inf.flowperExteriorSurfaceArea().is_initialized():
            program_type.infiltration = infiltration_from_openstudio(os_inf, schedules)
    # assign ventilation
    if os_space_type.designSpecificationOutdoorAir().is_initialized():
        os_vent = os_space_type.designSpecificationOutdoorAir().get()
        program_type.ventilation = ventilation_from_openstudio(os_vent, schedules)
    # assign the display name and return it
    if os_space_type.displayName().is_initialized():
        program_type.display_name = os_space_type.displayName().get()
    return program_type
