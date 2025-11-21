# coding=utf-8
"""OpenStudio construction translators."""
from __future__ import division
import re

from honeybee.typing import clean_ep_string
from honeybee_energy.material.shade import EnergyWindowMaterialShade, \
    EnergyWindowMaterialBlind
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.construction.windowshade import WindowConstructionShade
from honeybee_energy.construction.dynamic import WindowConstructionDynamic
from honeybee_energy.construction.shade import ShadeConstruction
from honeybee_energy.construction.air import AirBoundaryConstruction
from honeybee_energy.lib.constructions import air_boundary

from honeybee_openstudio.material import material_to_openstudio, extract_all_materials
from honeybee_openstudio.openstudio import OSConstruction, OSMaterialVector, \
    OSShadingControl, OSConstructionAirBoundary, OSZoneMixing, \
    OSStandardOpaqueMaterial, OSStandardGlazing, \
    OSOutputVariable, OSEnergyManagementSystemProgram, \
    OSEnergyManagementSystemSensor, OSEnergyManagementSystemActuator, \
    OSEnergyManagementSystemConstructionIndexVariable, os_vector_len


"""____________TRANSLATORS TO OPENSTUDIO____________"""


def standard_construction_to_openstudio(construction, os_model):
    """Convert Honeybee OpaqueConstruction or WindowConstruction to OpenStudio."""
    os_construction = OSConstruction(os_model)
    os_construction.setName(construction.identifier)
    if construction._display_name is not None:
        os_construction.setDisplayName(construction.display_name)
    os_materials = OSMaterialVector()
    for mat_id in construction.layers:
        material = os_model.getMaterialByName(mat_id)
        if material.is_initialized():
            os_material = material.get()
            try:
                os_materials.append(os_material)
            except AttributeError:  # using OpenStudio .NET bindings
                os_materials.Add(os_material)
    os_construction.setLayers(os_materials)
    return os_construction


def window_shade_construction_to_openstudio(construction, os_model):
    """Convert Honeybee WindowConstructionShade to OpenStudio Constructions."""
    # create the unshaded construction
    standard_construction_to_openstudio(construction.window_construction, os_model)
    # create the shaded construction
    os_shaded_con = OSConstruction(os_model)
    os_shaded_con.setName(construction.identifier)
    if construction._display_name is not None:
        os_shaded_con.setDisplayName(construction.display_name)
    os_materials = OSMaterialVector()
    for mat in construction.materials:
        material = os_model.getMaterialByName(mat.identifier)
        if material.is_initialized():
            os_material = material.get()
        else:  # it's a custom gap material that has not been added yet
            os_material = material_to_openstudio(mat, os_model)
        try:
            os_materials.append(os_material)
        except AttributeError:  # using OpenStudio .NET bindings
            os_materials.Add(os_material)
    os_shaded_con.setLayers(os_materials)
    return os_shaded_con


def window_shading_control_to_openstudio(construction, os_model):
    """Convert Honeybee WindowConstructionShade to OpenStudio ShadingControl.

    Each Aperture or Door that has a WindowConstructionShade assigned to it
    will have to call this method and then add the shading control to the
    OpenStudio SubSurface using the setShadingControl method.
    """
    # create the ShadingControl object
    os_shaded_con = os_model.getConstructionByName(construction.identifier)
    if os_shaded_con.is_initialized():
        os_shaded_con = os_shaded_con.get()
    else:
        msg = 'Failed to find construction "{}" for OpenStudio ShadingControl.'.format(
            construction.identifier)
        raise ValueError(msg)
    os_shade_control = OSShadingControl(os_shaded_con)
    # set the properties of the ShadingControl
    control_type = 'OnIfScheduleAllows' if construction.schedule is not None and \
        construction.control_type == 'AlwaysOn' else construction.control_type
    os_shade_control.setShadingControlType(control_type)
    os_shade_control.setShadingType(construction._ep_shading_type)
    if construction.schedule is not None:
        sch = os_model.getScheduleByName(construction.schedule.identifier)
        if sch.is_initialized():
            sch = sch.get()
            os_shade_control.setSchedule(sch)
    if construction.setpoint is not None:
        os_shade_control.setSetpoint(construction.setpoint)
    return os_shade_control


def window_dynamic_construction_to_openstudio(construction, os_model):
    """Convert Honeybee WindowConstructionDynamic to OpenStudio Constructions."""
    # write all states of the window constructions into the model
    os_constructions = []
    for i, con in enumerate(construction.constructions):
        con_dup = con.duplicate()
        con_dup.identifier = '{}State{}'.format(con.identifier, i)
        os_con = standard_construction_to_openstudio(con, os_model)
        os_constructions.append(os_con)
        os_con_i = OSEnergyManagementSystemConstructionIndexVariable(os_model, os_con)
        state_id = 'State{}{}'.format(i, re.sub('[^A-Za-z0-9]', '', con.identifier))
        os_con_i.setName(state_id)
    # set up the EMS sensor for the schedule value
    sensor_id = 'Sensor{}'.format(re.sub('[^A-Za-z0-9]', '', construction.identifier))
    schedule_id = construction.schedule.identifier
    state_sch = os_model.getScheduleByName(schedule_id)
    if state_sch.is_initialized():
        sch_var = OSOutputVariable('Schedule Value', os_model)
        sch_var.setReportingFrequency('Timestep')
        sch_var.setKeyValue(schedule_id)
        sch_sens = OSEnergyManagementSystemSensor(os_model, sch_var)
        sch_sens.setName(sensor_id)
    return os_constructions


def window_dynamic_ems_program_to_openstudio(construction, os_sub_faces, os_model):
    """Convert WindowConstructionDynamic to OpenStudio EnergyManagementSystemProgram.

    Args:
        construction: A honeybee-energy WindowConstructionDynamic for which an
            EnergyManagementSystemProgram will be written.
        os_sub_faces: A list of OpenStudio SubSurface objects for all of the Apertures
            and Doors that have the dynamic construction assigned to them.
        os_model: The OpenStudio Model to which the dynamic window construction
            is being added.
    """
    # create the actuators
    actuator_ids = []
    for i, os_sf in enumerate(os_sub_faces):
        window_act = OSEnergyManagementSystemActuator(
            os_sf, 'Surface', 'Construction State')
        ap_id = os_sf.nameString()
        act_id = 'Actuator{}{}'.format(i, re.sub('[^A-Za-z0-9]', '', ap_id))
        window_act.setName(act_id)
        actuator_ids.append(act_id)
    # get the lines of the EMS program add each construction state to the program
    ems_program = []
    sensor_id = 'Sensor{}'.format(re.sub('[^A-Za-z0-9]', '', construction.identifier))
    max_state_count = len(construction.constructions) - 1
    for i, con in enumerate(construction.constructions):
        # determine which conditional operator to use
        cond_op = 'IF' if i == 0 else 'ELSEIF'
        # add the conditional statement
        state_count = i + 1
        if i == max_state_count:
            cond_stmt = 'ELSE'
        else:
            cond_stmt = '{} ({} < {})'.format(cond_op, sensor_id, state_count)
        ems_program.append(cond_stmt)
        # loop through the actuators and set the appropriate window state
        state_id = 'State{}{}'.format(i, re.sub('[^A-Za-z0-9]', '', con.identifier))
        for act_name in actuator_ids:
            ems_program.append('SET {} = {}'.format(act_name, state_id))
    ems_program.append('ENDIF')
    # create the EMS Program object
    os_ems_prog = OSEnergyManagementSystemProgram(os_model)
    pid = 'StateChange{}'.format(re.sub('[^A-Za-z0-9]', '', construction.identifier))
    os_ems_prog.setName(pid)
    for line in ems_program:
        os_ems_prog.addLine(line)
    return os_ems_prog


def air_construction_to_openstudio(construction, os_model):
    """Convert Honeybee AirBoundaryConstruction to OpenStudio ConstructionAirBoundary."""
    os_construction = OSConstructionAirBoundary(os_model)
    os_construction.setName(construction.identifier)
    if construction._display_name is not None:
        os_construction.setDisplayName(construction.display_name)
    os_construction.setAirExchangeMethod('None')
    return os_construction


def air_mixing_to_openstudio(face, target_zone, source_zone, os_model):
    """Convert Honeybee AirBoundaryConstruction to OpenStudio ZoneMixing.

    Args:
        face: A honeybee Face that has an AirBoundary face type.
        target_zone: The OpenStudio ThermalZone for the target of air mixing.
        source_zone: The OpenStudio ThermalZone for the source of air mixing.
        os_model: The OpenStudio Model to which the zone mixing is being added.
    """
    # calculate the flow rate and schedule
    construction = face.properties.energy.construction
    if isinstance(construction, AirBoundaryConstruction):
        flow_rate = face.area * construction.air_mixing_per_area
        schedule = construction.air_mixing_schedule.identifier
    else:
        flow_rate = face.area * air_boundary.air_mixing_per_area
        schedule = air_boundary.air_mixing_schedule.identifier
    # create the ZoneMixing object
    os_zone_mixing = OSZoneMixing(target_zone)
    os_zone_mixing.setSourceZone(source_zone)
    os_zone_mixing.setDesignFlowRate(flow_rate)
    flow_sch_ref = os_model.getScheduleByName(schedule)
    if flow_sch_ref.is_initialized():
        flow_sched = flow_sch_ref.get()
        os_zone_mixing.setSchedule(flow_sched)
    return os_zone_mixing


def shade_construction_to_openstudio(construction, os_model):
    """Convert Honeybee ShadeConstruction to OpenStudio Construction."""
    os_construction = OSConstruction(os_model)
    os_construction.setName(construction.identifier)
    if construction._display_name is not None:
        os_construction.setDisplayName(construction.display_name)
    os_materials = OSMaterialVector()
    if construction.is_specular:
        os_material = OSStandardGlazing(os_model)
        os_material.setFrontSideSolarReflectanceatNormalIncidence(
            construction.solar_reflectance)
        os_material.setFrontSideVisibleReflectanceatNormalIncidence(
            construction.visible_reflectance)
        os_material.setSolarTransmittanceatNormalIncidence(0)
        os_material.setVisibleTransmittanceatNormalIncidence(0)
    else:
        os_material = OSStandardOpaqueMaterial(os_model)
        os_material.setSolarAbsorptance(1 - construction.solar_reflectance)
        os_material.setVisibleAbsorptance(1 - construction.visible_reflectance)
    try:
        os_materials.append(os_material)
    except AttributeError:  # using OpenStudio .NET bindings
        os_materials.Add(os_material)
    os_construction.setLayers(os_materials)
    return os_construction


def construction_to_openstudio(construction, os_model):
    """Convert any Honeybee energy construction into an OpenStudio object.

    Args:
        construction: A honeybee-energy Python object of a construction.
        os_model: The OpenStudio Model object to which the Room will be added.

    Returns:
        An OpenStudio object for the construction.
    """
    if isinstance(construction, (OpaqueConstruction, WindowConstruction)):
        return standard_construction_to_openstudio(construction, os_model)
    elif isinstance(construction, WindowConstructionShade):
        return window_shade_construction_to_openstudio(construction, os_model)
    elif isinstance(construction, WindowConstructionDynamic):
        return window_dynamic_construction_to_openstudio(construction, os_model)
    elif isinstance(construction, ShadeConstruction):
        return shade_construction_to_openstudio(construction, os_model)
    elif isinstance(construction, AirBoundaryConstruction):
        return air_construction_to_openstudio(construction, os_model)
    else:
        raise ValueError(
            '{} is not a recognized Energy Construction type'.format(type(construction))
        )


"""____________TRANSLATORS FROM OPENSTUDIO____________"""


def opaque_construction_from_openstudio(os_construction, materials):
    """Convert OpenStudio Construction to Honeybee OpaqueConstruction."""
    layers = []
    for layer in os_construction.layers():
        try:
            layers.append(materials[clean_ep_string(layer.nameString())])
        except KeyError:  # material that could not be serialized
            return None
    construct = OpaqueConstruction(clean_ep_string(os_construction.nameString()), layers)
    if os_construction.displayName().is_initialized():
        construct.display_name = os_construction.displayName().get()
    return construct


def window_construction_from_openstudio(os_construction, materials):
    """Convert OpenStudio Construction to Honeybee WindowConstruction."""
    layers, skip = [], False
    all_layers = os_construction.layers()
    for i, layer in enumerate(all_layers):
        if skip:
            skip = False
            continue
        try:
            mat = materials[clean_ep_string(layer.nameString())]
        except KeyError:  # material that could not be serialized
            return None
        if not isinstance(mat, (EnergyWindowMaterialShade, EnergyWindowMaterialBlind)):
            layers.append(mat)
        elif i not in (0, os_vector_len(all_layers) - 1):
            skip = True
    construct = WindowConstruction(clean_ep_string(os_construction.nameString()), layers)
    if os_construction.displayName().is_initialized():
        construct.display_name = os_construction.displayName().get()
    return construct


def air_construction_from_openstudio(os_construction, schedules=None):
    """Convert OpenStudio ConstructionAirBoundary to Honeybee AirBoundaryConstruction."""
    construct = AirBoundaryConstruction(clean_ep_string(os_construction.nameString()))
    if schedules is not None and os_construction.simpleMixingSchedule().is_initialized():
        schedule = os_construction.simpleMixingSchedule().get()
        try:
            construct.air_mixing_schedule = \
                schedules[clean_ep_string(schedule.nameString())]
        except KeyError:  # schedule that could not be serialized
            pass
    if os_construction.displayName().is_initialized():
        construct.display_name = os_construction.displayName().get()
    return construct


def shade_construction_from_openstudio(os_construction):
    """Convert OpenStudio Construction to Honeybee OpaqueConstruction."""
    solar_ref, visible_ref, is_specular = 0.2, 0.2, False
    layer = os_construction.layers()[0]
    if layer.to_StandardGlazing().is_initialized():
        layer = layer.to_StandardGlazing().get()
        is_specular = True
        if layer.frontSideSolarReflectanceatNormalIncidence().is_initialized():
            solar_ref = layer.frontSideSolarReflectanceatNormalIncidence().get()
        if layer.frontSideVisibleReflectanceatNormalIncidence().is_initialized():
            visible_ref = layer.frontSideVisibleReflectanceatNormalIncidence().get()
    elif layer.to_StandardOpaqueMaterial().is_initialized():
        layer = layer.to_StandardOpaqueMaterial().get()
        is_specular = False
        if layer.solarReflectance().is_initialized():
            solar_ref = layer.solarReflectance().get()
        if layer.visibleReflectance().is_initialized():
            visible_ref = layer.visibleReflectance().get()
    elif layer.to_MasslessOpaqueMaterial().is_initialized():
        layer = layer.to_MasslessOpaqueMaterial().get()
        is_specular = False
        if layer.solarAbsorptance().is_initialized():
            solar_ref = 1 - layer.solarAbsorptance().get()
        if layer.visibleAbsorptance().is_initialized():
            visible_ref = 1 - layer.visibleAbsorptance().get()
    const_id = '{} Shade'.format(clean_ep_string(os_construction.nameString()))
    construct = ShadeConstruction(const_id, solar_ref, visible_ref, is_specular)
    if os_construction.displayName().is_initialized():
        construct.display_name = os_construction.displayName().get()
    return construct


def extract_all_constructions(os_model, schedules=None):
    """Extract all construction objects from an OpenStudio Model.

    Args:
        os_model: The OpenStudio Model object from which constructions will be extracted.
        materials: An optional dictionary of schedules with schedule identifiers
            as keys and schedules objects as values.

    Returns:
        A dictionary of construction objects with construction identifiers as keys and
        construction objects as values.
    """
    # extract all of the materials from the model
    materials = extract_all_materials(os_model)
    constructions = {}
    # create HB AirConstruction from OpenStudio Construction
    for os_construction in os_model.getConstructionAirBoundarys():
        construct = air_construction_from_openstudio(os_construction, schedules)
        constructions[construct.identifier] = construct
    # create HB WindowConstruction from OpenStudio Construction
    for os_construction in os_model.getConstructions():
        layers = os_construction.layers()
        if os_vector_len(layers) > 0:
            window_construction, opaque_construction = False, False
            material = layers[0]
            if material.to_StandardGlazing().is_initialized() or \
                    material.to_SimpleGlazing().is_initialized():
                window_construction = True
            elif material.to_StandardOpaqueMaterial().is_initialized() or \
                    material.to_MasslessOpaqueMaterial().is_initialized() or \
                    material.to_RoofVegetation().is_initialized():
                opaque_construction = True
            if window_construction:
                constr = window_construction_from_openstudio(os_construction, materials)
                if constr is not None:
                    constructions[constr.identifier] = constr
            elif opaque_construction:
                constr = opaque_construction_from_openstudio(os_construction, materials)
                if constr is not None:
                    constructions[constr.identifier] = constr
    return constructions
