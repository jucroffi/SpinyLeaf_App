# coding=utf-8
"""Methods to write Honeybee Models to OpenStudio."""
from __future__ import division
import sys
import os
import tempfile
import json
import subprocess
import xml.etree.ElementTree as ET

from ladybug_geometry.geometry3d import Face3D
from honeybee.typing import clean_ep_string
from honeybee.altnumber import autocalculate
from honeybee.facetype import RoofCeiling, Floor, AirBoundary
from honeybee.boundarycondition import Outdoors, Surface
from honeybee.model import Model
from honeybee_energy.config import folders as hbe_folders
from honeybee_energy.boundarycondition import Adiabatic, OtherSideTemperature
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.construction.windowshade import WindowConstructionShade
from honeybee_energy.construction.dynamic import WindowConstructionDynamic
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.hvac._template import _TemplateSystem
from honeybee_energy.hvac.detailed import DetailedHVAC
from honeybee_energy.lib.constructionsets import generic_construction_set

from honeybee_openstudio.openstudio import OSModel, OSPoint3dVector, OSPoint3d, \
    OSShadingSurfaceGroup, OSShadingSurface, OSSubSurface, OSSurface, OSSpace, \
    OSThermalZone, OSBuildingStory, OSSurfacePropertyOtherSideCoefficients, \
    OSEnergyManagementSystemProgramCallingManager, openstudio, os_path, os_vector_len
from honeybee_openstudio.schedule import schedule_type_limits_to_openstudio, \
    schedule_to_openstudio
from honeybee_openstudio.material import material_to_openstudio
from honeybee_openstudio.construction import construction_to_openstudio, \
    air_mixing_to_openstudio, window_shading_control_to_openstudio, \
    window_dynamic_ems_program_to_openstudio
from honeybee_openstudio.constructionset import construction_set_to_openstudio
from honeybee_openstudio.internalmass import internal_mass_to_openstudio
from honeybee_openstudio.load import people_to_openstudio, lighting_to_openstudio, \
    electric_equipment_to_openstudio, gas_equipment_to_openstudio, \
    hot_water_to_openstudio, process_to_openstudio, \
    infiltration_to_openstudio, ventilation_to_openstudio, \
    setpoint_to_openstudio_thermostat, setpoint_to_openstudio_humidistat, \
    daylight_to_openstudio
from honeybee_openstudio.programtype import program_type_to_openstudio
from honeybee_openstudio.ventcool import ventilation_opening_to_openstudio, \
    ventilation_fan_to_openstudio, ventilation_sim_control_to_openstudio, \
    afn_crack_to_openstudio, ventilation_opening_to_openstudio_afn, \
    ventilation_control_to_openstudio_afn, zone_temperature_sensor, \
    outdoor_temperature_sensor, ventilation_control_program_manager
from honeybee_openstudio.shw import shw_system_to_openstudio
from honeybee_openstudio.hvac.idealair import ideal_air_system_to_openstudio
from honeybee_openstudio.hvac.template import template_hvac_to_openstudio
from honeybee_openstudio.generator import pv_properties_to_openstudio, \
    electric_load_center_to_openstudio
from honeybee_openstudio.hvac.standards.utilities import \
    rename_air_loop_nodes, rename_plant_loop_nodes


def face_3d_to_openstudio(face_3d):
    """Convert a Face3D into an OpenStudio Point3dVector.

    Args:
        face_3d: A ladybug-geometry Face3D object for which an OpenStudio Point3dVector
            string will be generated.

    Returns:
        An OpenStudio Point3dVector to be used to construct geometry objects.
    """
    os_vertices = OSPoint3dVector()
    for pt in face_3d.upper_left_counter_clockwise_vertices:
        try:
            os_vertices.append(OSPoint3d(pt.x, pt.y, pt.z))
        except AttributeError:  # using OpenStudio .NET bindings
            os_vertices.Add(OSPoint3d(pt.x, pt.y, pt.z))
    return os_vertices


def shade_mesh_to_openstudio(shade_mesh, os_model):
    """Create OpenStudio objects from a ShadeMesh.

    Args:
        shade_mesh: A honeybee ShadeMesh for which OpenStudio objects will be returned.
        os_model: The OpenStudio Model object to which the ShadeMesh will be added.

    Returns:
        A list of OpenStudio ShadingSurface objects.
    """
    # loop through the mesh faces and create individual shade objects
    os_shades = []
    os_shd_group = OSShadingSurfaceGroup(os_model)
    os_shd_group.setName(shade_mesh.identifier)
    for i, shade in enumerate(shade_mesh.geometry.face_vertices):
        # create the shade object with the geometry
        shade_face = Face3D(shade)
        os_vertices = face_3d_to_openstudio(shade_face)
        os_shade = OSShadingSurface(os_vertices, os_model)
        os_shade.setName('{}_{}'.format(shade_mesh.identifier, i))
        os_shade.setShadingSurfaceGroup(os_shd_group)
        os_shades.append(os_shade)
    if shade_mesh._display_name is not None:
        for os_shade in os_shades:
            os_shade.setDisplayName(shade_mesh.display_name)
    # assign the construction and transmittance
    construction = shade_mesh.properties.energy.construction
    if shade_mesh.properties.energy.is_construction_set_on_object and \
            not construction.is_default:
        os_construction = os_model.getConstructionByName(construction.identifier)
        if os_construction.is_initialized():
            os_construction = os_construction.get()
            for os_shade in os_shades:
                os_shade.setConstruction(os_construction)
    trans_sched = shade_mesh.properties.energy.transmittance_schedule
    if trans_sched is not None:
        os_schedule = os_model.getScheduleByName(trans_sched.identifier)
        if os_schedule.is_initialized():
            os_schedule = os_schedule.get()
            for os_shade in os_shades:
                os_shade.setTransmittanceSchedule(os_schedule)
    return os_shades


def shade_to_openstudio(shade, os_model):
    """Create an OpenStudio object from a Shade.

    Args:
        shade: A honeybee Shade for which an OpenStudio object will be returned.
        os_model: The OpenStudio Model object to which the Shade will be added.

    Returns:
        An OpenStudio ShadingSurface object.
    """
    # create the shade object with the geometry
    os_vertices = face_3d_to_openstudio(shade.geometry)
    os_shade = OSShadingSurface(os_vertices, os_model)
    os_shade.setName(shade.identifier)
    if shade._display_name is not None:
        os_shade.setDisplayName(shade.display_name)
    # assign the construction and transmittance
    construction = shade.properties.energy.construction
    if shade.properties.energy.is_construction_set_on_object and \
            not construction.is_default:
        os_construction = os_model.getConstructionByName(construction.identifier)
        if os_construction.is_initialized():
            os_construction = os_construction.get()
            os_shade.setConstruction(os_construction)
    trans_sched = shade.properties.energy.transmittance_schedule
    if trans_sched is not None:
        os_schedule = os_model.getScheduleByName(trans_sched.identifier)
        if os_schedule.is_initialized():
            os_schedule = os_schedule.get()
            os_shade.setTransmittanceSchedule(os_schedule)
    # add the PVProperties if they exist
    pv_prop = shade.properties.energy.pv_properties
    if pv_prop is not None:
        pv_properties_to_openstudio(pv_prop, os_shade, os_model)
    return os_shade


def door_to_openstudio(door, os_model):
    """Create an OpenStudio object from a Door.

    Args:
        door: A honeybee Door for which an OpenStudio object will be returned.
        os_model: The OpenStudio Model object to which the Door will be added.

    Returns:
        An OpenStudio SubSurface object if the Door has a parent. An OpenStudio
        ShadingSurface object if the Door has no parent.
    """
    # convert the base geometry to OpenStudio
    os_vertices = face_3d_to_openstudio(door.geometry)
    # translate the geometry to either a SubSurface or a ShadingSurface
    if door.has_parent:  # translate the geometry as SubSurface
        os_door = OSSubSurface(os_vertices, os_model)
        if door.is_glass:
            dr_type = 'GlassDoor'
        else:
            par = door.parent
            dr_type = 'OverheadDoor' if isinstance(par.boundary_condition, Outdoors) and \
                isinstance(par.type, (RoofCeiling, Floor)) else 'Door'
        os_door.setSubSurfaceType(dr_type)
        # assign the construction if it's hard set
        construction = door.properties.energy.construction
        if door.properties.energy.is_construction_set_on_object:
            if construction.has_shade:
                constr_id = construction.window_construction.identifier
            elif construction.is_dynamic:
                constr_id = '{}State0'.format(construction.constructions[0].identifier)
            else:
                constr_id = construction.identifier
            os_construction = os_model.getConstructionByName(constr_id)
            if os_construction.is_initialized():
                os_construction = os_construction.get()
                os_door.setConstruction(os_construction)
        # assign the frame property if the window construction has one
        if construction.has_frame:
            frame_id = construction.frame.identifier
            frame = os_model.getWindowPropertyFrameAndDividerByName(frame_id)
            if frame.is_initialized():
                os_frame = frame.get()
                os_door.setWindowPropertyFrameAndDivider(os_frame)
        # create the WindowShadingControl object if it is needed
        if construction.has_shade:
            shd_prop_str = window_shading_control_to_openstudio(construction, os_model)
            os_door.setShadingControl(shd_prop_str)
    else:  # translate the geometry as ShadingSurface
        os_door = OSShadingSurface(os_vertices, os_model)
        cns = door.properties.energy.construction
        os_construction = os_model.getConstructionByName(cns.identifier)
        if os_construction.is_initialized():
            os_construction = os_construction.get()
            os_door.setConstruction(os_construction)
        if door.is_glass:
            trans_sch = 'Constant %.3f Transmittance' % cns.solar_transmittance
            os_schedule = os_model.getScheduleByName(trans_sch)
            if os_schedule.is_initialized():
                os_schedule = os_schedule.get()
                os_door.setTransmittanceSchedule(os_schedule)
        # translate any shades assigned to the Door
        for shd in door._outdoor_shades:
            shade_to_openstudio(shd, os_model)

    # set the object name and return it
    os_door.setName(door.identifier)
    if door._display_name is not None:
        os_door.setDisplayName(door.display_name)
    return os_door


def aperture_to_openstudio(aperture, os_model):
    """Create an OpenStudio object from an Aperture.

    Args:
        aperture: A honeybee Aperture for which an OpenStudio object will be returned.
        os_model: The OpenStudio Model object to which the Aperture will be added.

    Returns:
        An OpenStudio SubSurface object if the Aperture has a parent. An OpenStudio
        ShadingSurface object if the Aperture has no parent.
    """
    # convert the base geometry to OpenStudio
    os_vertices = face_3d_to_openstudio(aperture.geometry)
    # translate the geometry to either a SubSurface or a ShadingSurface
    if aperture.has_parent:  # translate the geometry as SubSurface
        os_aperture = OSSubSurface(os_vertices, os_model)
        if aperture.is_operable:
            ap_type = 'OperableWindow'
        else:
            par = aperture.parent
            ap_type = 'Skylight' if isinstance(par.boundary_condition, Outdoors) and \
                isinstance(par.type, (RoofCeiling, Floor)) else 'FixedWindow'
        os_aperture.setSubSurfaceType(ap_type)
        # assign the construction if it's hard set
        construction = aperture.properties.energy.construction
        if aperture.properties.energy.is_construction_set_on_object:
            if construction.has_shade:
                constr_id = construction.window_construction.identifier
            elif construction.is_dynamic:
                constr_id = '{}State0'.format(construction.constructions[0].identifier)
            else:
                constr_id = construction.identifier
            os_construction = os_model.getConstructionByName(constr_id)
            if os_construction.is_initialized():
                os_construction = os_construction.get()
                os_aperture.setConstruction(os_construction)
        # assign the frame property if the window construction has one
        if construction.has_frame:
            frame_id = construction.frame.identifier
            frame = os_model.getWindowPropertyFrameAndDividerByName(frame_id)
            if frame.is_initialized():
                os_frame = frame.get()
                os_aperture.setWindowPropertyFrameAndDivider(os_frame)
        # create the WindowShadingControl object if it is needed
        if construction.has_shade:
            shd_prop_str = window_shading_control_to_openstudio(construction, os_model)
            os_aperture.setShadingControl(shd_prop_str)
    else:  # translate the geometry as ShadingSurface
        os_aperture = OSShadingSurface(os_vertices, os_model)
        cns = aperture.properties.energy.construction
        os_construction = os_model.getConstructionByName(cns.identifier)
        if os_construction.is_initialized():
            os_construction = os_construction.get()
            os_aperture.setConstruction(os_construction)
        trans_sch = 'Constant %.3f Transmittance' % cns.solar_transmittance
        os_schedule = os_model.getScheduleByName(trans_sch)
        if os_schedule.is_initialized():
            os_schedule = os_schedule.get()
            os_aperture.setTransmittanceSchedule(os_schedule)
        # translate any shades assigned to the Aperture
        for shd in aperture._outdoor_shades:
            shade_to_openstudio(shd, os_model)

    # set the object name and return it
    os_aperture.setName(aperture.identifier)
    if aperture._display_name is not None:
        os_aperture.setDisplayName(aperture.display_name)
    return os_aperture


def face_to_openstudio(face, os_model, adj_map=None, ignore_complex_sub_faces=True):
    """Create an OpenStudio object from a Face.

    This method also adds all Apertures, Doors, and Shades assigned to the Face.

    Args:
        face: A honeybee Face for which an OpenStudio object will be returned.
        os_model: The OpenStudio Model object to which the Face will be added.
        adj_map: An optional dictionary with keys for 'faces' and 'sub_faces'
            that will have the space Surfaces and SubSurfaces added to it
            such that adjacencies can be assigned after running this method.
        ignore_complex_sub_faces: Boolean for whether sub-faces (including Apertures
            and Doors) should be ignored if they have more than 4 sides (True) or
            whether they should be left as they are (False). (Default: True).

    Returns:
        An OpenStudio Surface object if the Face has a parent. An OpenStudio
        ShadingSurface object if the Face has no parent.
    """
    # translate the geometry to either a SubSurface or a ShadingSurface
    if face.has_parent:
        # create the Surface
        os_vertices = face_3d_to_openstudio(face.geometry)
        os_face = OSSurface(os_vertices, os_model)

        # select the correct face type
        if isinstance(face.type, AirBoundary):
            os_f_type = 'Wall'  # air boundaries are not a Surface type in EnergyPlus
        elif isinstance(face.type, RoofCeiling):
            if face.altitude < 0:
                os_f_type = 'Wall'  # ensure E+ does not try to flip the Face
            else:
                os_f_type = 'RoofCeiling'
        elif isinstance(face.type, Floor) and face.altitude > 0:
            os_f_type = 'Wall'  # ensure E+ does not try to flip the Face
        else:
            os_f_type = face.type.name
        os_face.setSurfaceType(os_f_type)

        # assign the boundary condition
        fbc = face.boundary_condition
        if not isinstance(fbc, (Surface, OtherSideTemperature)):
            os_face.setOutsideBoundaryCondition(fbc.name)
        if isinstance(fbc, Outdoors):
            if not fbc.sun_exposure:
                os_face.setSunExposure('NoSun')
            if not fbc.wind_exposure:
                os_face.setWindExposure('NoWind')
            if fbc.view_factor != autocalculate:
                os_face.setViewFactortoGround(fbc.view_factor)
        elif isinstance(fbc, OtherSideTemperature):
            srf_prop = OSSurfacePropertyOtherSideCoefficients(os_model)
            srf_prop.setName('{}_OtherTemp'.format(face.identifier))
            htc = fbc.heat_transfer_coefficient
            srf_prop.setCombinedConvectiveRadiativeFilmCoefficient(htc)
            if fbc.temperature == autocalculate:
                srf_prop.setConstantTemperatureCoefficient(0)
                srf_prop.setExternalDryBulbTemperatureCoefficient(1)
            else:
                srf_prop.setConstantTemperature(fbc.temperature)
                srf_prop.setConstantTemperatureCoefficient(1)
                srf_prop.setExternalDryBulbTemperatureCoefficient(0)
            os_face.setSurfacePropertyOtherSideCoefficients(srf_prop)

        # assign the construction if it's hard set, an AirBoundary, or Adiabatic
        if face.properties.energy.is_construction_set_on_object or \
                isinstance(face.type, AirBoundary) or \
                isinstance(face.boundary_condition, (Adiabatic, OtherSideTemperature)):
            construction_id = face.properties.energy.construction.identifier
            os_construction = os_model.getConstructionByName(construction_id)
            if os_construction.is_initialized():
                os_construction = os_construction.get()
                os_face.setConstruction(os_construction)

        # create the sub-faces
        sub_faces = {}
        for ap in face.apertures:
            # ignore apertures to be triangulated
            if len(ap.geometry) <= 4 or not ignore_complex_sub_faces:
                os_ap = aperture_to_openstudio(ap, os_model)
                os_ap.setSurface(os_face)
                sub_faces[ap.identifier] = os_ap
        for dr in face.doors:
            # ignore doors to be triangulated
            if len(dr.geometry) <= 4 or not ignore_complex_sub_faces:
                os_dr = door_to_openstudio(dr, os_model)
                os_dr.setSurface(os_face)
                sub_faces[dr.identifier] = os_dr

        # update the adjacency map if it exists
        if adj_map is not None:
            adj_map['faces'][face.identifier] = os_face
            adj_map['sub_faces'].update(sub_faces)
    else:
        os_vertices = face_3d_to_openstudio(face.punched_geometry)
        os_face = OSShadingSurface(os_vertices, os_model)
        for ap in face.apertures:
            aperture_to_openstudio(ap.duplicate(), os_model)
        for dr in face.doors:
            door_to_openstudio(dr.duplicate(), os_model)
        for shd in face._outdoor_shades:
            shade_to_openstudio(shd, os_model)

    # set the object name and return it
    os_face.setName(face.identifier)
    if face._display_name is not None:
        os_face.setDisplayName(face.display_name)
    return os_face


def room_to_openstudio(room, os_model, adj_map=None, include_infiltration=True,
                       ignore_complex_sub_faces=True):
    """Create OpenStudio objects from a Room.

    Args:
        room: A honeybee Room for which an OpenStudio object will be returned.
        os_model: The OpenStudio Model object to which the Room will be added.
        adj_map: An optional dictionary with keys for 'faces' and 'sub_faces'
            that will have the space Surfaces and SubSurfaces added to it
            such that adjacencies can be assigned after running this method.
        include_infiltration: Boolean for whether or not infiltration will be included
            in the translation of the Room. It may be desirable to set this
            to False if the building airflow is being modeled with the EnergyPlus
            AirFlowNetwork. (Default: True).
        ignore_complex_sub_faces: Boolean for whether sub-faces (including Apertures
            and Doors) should be ignored if they have more than 4 sides (True) or
            whether they should be left as they are (False). (Default: True).

    Returns:
        An OpenStudio Space object for the Room.
    """
    # create the space
    os_space = OSSpace(os_model)
    os_space.setName('{}_Space'.format(room.identifier))
    if room._display_name is not None:
        os_space.setDisplayName(room.display_name)
    if room.exclude_floor_area:
        if sys.version_info < (3, 0):  # .NET bindings are missing the method
            os_space.setString(11, 'No')
        else:
            os_space.setPartofTotalFloorArea(False)
    try:
        os_space.setVolume(room.volume)
    except AttributeError:  # older OpenStudio bindings where method was not implemented
        pass

    # assign the construction set if specified
    if room.properties.energy._construction_set is not None:
        con_set_id = room.properties.energy.construction_set.identifier
        os_con_set = os_model.getDefaultConstructionSetByName(con_set_id)
        if os_con_set.is_initialized():
            os_con_set = os_con_set.get()
            os_space.setDefaultConstructionSet(os_con_set)

    # assign the program type if specified
    overridden_loads = room.properties.energy.has_overridden_space_loads
    if not overridden_loads and room.properties.energy._program_type is not None:
        # assign loads using the OpenStudio SpaceType
        space_type_id = room.properties.energy.program_type.identifier
        os_space_type = os_model.getSpaceTypeByName(space_type_id)
        if os_space_type.is_initialized():
            space_type_object = os_space_type.get()
            os_space.setSpaceType(space_type_object)
    elif overridden_loads:
        # assign loads directly to the space
        people = room.properties.energy.people
        if people is not None:
            os_people = people_to_openstudio(people, os_model)
            os_people.setName('{}..{}'.format(people.identifier, room.identifier))
            os_people.setSpace(os_space)
        lighting = room.properties.energy.lighting
        if lighting is not None:
            os_lights = lighting_to_openstudio(lighting, os_model)
            os_lights.setName('{}..{}'.format(lighting.identifier, room.identifier))
            os_lights.setSpace(os_space)
        equipment = room.properties.energy.electric_equipment
        if equipment is not None:
            os_equip = electric_equipment_to_openstudio(equipment, os_model)
            os_equip.setName('{}..{}'.format(equipment.identifier, room.identifier))
            os_equip.setSpace(os_space)
        equipment = room.properties.energy.gas_equipment
        if equipment is not None:
            os_equip = gas_equipment_to_openstudio(equipment, os_model)
            os_equip.setName('{}..{}'.format(equipment.identifier, room.identifier))
            os_equip.setSpace(os_space)
        infilt = room.properties.energy.infiltration
        if infilt is not None and include_infiltration:
            os_inf = infiltration_to_openstudio(infilt, os_model)
            os_inf.setName('{}..{}'.format(infilt.identifier, room.identifier))
            os_inf.setSpace(os_space)
    # assign the ventilation and catch the case that the SpaceType one is not correct
    if overridden_loads or room.properties.energy._ventilation is not None:
        vent = room.properties.energy.ventilation
        if vent is not None:
            os_vent = os_model.getDesignSpecificationOutdoorAirByName(vent.identifier)
            if os_vent.is_initialized():
                os_vent = os_vent.get()
            else:
                os_vent = ventilation_to_openstudio(vent, os_model)
            os_space.setDesignSpecificationOutdoorAir(os_vent)
    # assign all process loads
    for process in room.properties.energy.process_loads:
        os_process = process_to_openstudio(process, os_model)
        os_process.setName('{}..{}'.format(process.identifier, room.identifier))
        os_process.setSpace(os_space)
    # assign the daylight control if it is specified
    daylight = room.properties.energy.daylighting_control
    if daylight is not None:
        os_daylight = daylight_to_openstudio(daylight, os_model)
        os_daylight.setName('{}_Daylighting'.format(room.identifier))
        os_daylight.setSpace(os_space)
    # assign any internal mass definitions if specified
    for mass in room.properties.energy.internal_masses:
        os_mass = internal_mass_to_openstudio(mass, os_model)
        os_mass.setName('{}::{}'.format(mass.identifier, room.identifier))
        os_mass.setSpace(os_space)

    # assign all of the faces to the room
    for face in room.faces:
        os_face = face_to_openstudio(face, os_model, adj_map,
                                     ignore_complex_sub_faces=ignore_complex_sub_faces)
        os_face.setSpace(os_space)

    # add any assigned shades to a group for the room
    child_shades = []
    child_shades.extend(room._outdoor_shades)
    for face in room._faces:
        child_shades.extend(face._outdoor_shades)
        for ap in face.apertures:
            child_shades.extend(ap._outdoor_shades)
        for dr in face.doors:
            child_shades.extend(dr._outdoor_shades)
    if len(child_shades) != 0:
        os_shd_group = OSShadingSurfaceGroup(os_model)
        os_shd_group.setName('{} Shades'.format(room.identifier))
        os_shd_group.setSpace(os_space)
        os_shd_group.setShadingSurfaceType('Space')
        for shd in child_shades:
            os_shade = shade_to_openstudio(shd, os_model)
            os_shade.setShadingSurfaceGroup(os_shd_group)

    return os_space


def model_to_openstudio(
    model, seed_model=None, schedule_directory=None,
    use_geometry_names=False, use_resource_names=False,
    triangulate_non_planar_orphaned=False, triangulate_subfaces=True,
    use_simple_window_constructions=False, enforce_rooms=False, print_progress=False
):
    """Create an OpenStudio Model from a Honeybee Model.

    The resulting Model will include all geometry (Rooms, Faces, Apertures,
    Doors, Shades), all fully-detailed constructions + materials, all fully-detailed
    schedules, and the room properties.

    Args:
        model: The Honeybee Model to be converted into an OpenStudio Model.
        seed_model: An optional OpenStudio Model object to which the Honeybee
            Model will be added. If None, a new OpenStudio Model will be
            initialized within this method. (Default: None).
        schedule_directory: An optional file directory to which all file-based
            schedules should be written to. If None, all ScheduleFixedIntervals
            will be translated to Schedule:Compact and written fully into the
            IDF string instead of to Schedule:File. (Default: None).
        use_geometry_names: Boolean to note whether a cleaned version of all
            geometry display names should be used instead of identifiers when
            translating the Model to OSM and IDF. Using this flag will affect
            all Rooms, Faces, Apertures, Doors, and Shades. It will generally
            result in more read-able names in the OSM and IDF but this means
            that it will not be easy to map the EnergyPlus results back to the
            input Honeybee Model. Cases of duplicate IDs resulting from
            non-unique names will be resolved by adding integers to the ends
            of the new IDs that are derived from the name. (Default: False).
        use_resource_names: Boolean to note whether a cleaned version of all
            resource display names should be used instead of identifiers when
            translating the Model to OSM and IDF. Using this flag will affect
            all Materials, Constructions, ConstructionSets, Schedules, Loads,
            and ProgramTypes. It will generally result in more read-able names
            for the resources in the OSM and IDF. Cases of duplicate IDs
            resulting from non-unique names will be resolved by adding integers
            to the ends of the new IDs that are derived from the name. (Default: False).
        triangulate_non_planar_orphaned: Boolean to note whether any non-planar
            orphaned geometry in the model should be triangulated upon export.
            This can be helpful because OpenStudio simply raises an error when
            it encounters non-planar geometry, which would hinder the ability
            to save files that are to be corrected later. (Default: False).
        triangulate_subfaces: Boolean to note whether sub-faces (including
            Apertures and Doors) should be triangulated if they have more
            than 4 sides (True) or whether they should be left as they are (False).
            This triangulation is necessary when exporting directly to EnergyPlus
            since it cannot accept sub-faces with more than 4 vertices. (Default: True).
        use_simple_window_constructions: Boolean to note whether the Model should
            be translated with simple window constructions, all of which will
            be represented with a single-layer glazing system construction. This
            is useful for translation to gbXML since the U-value will only show
            up if the construction is simple. (Default: False).
        enforce_rooms: Boolean to note whether this method should enforce the
            presence of Rooms in the Model, which is as necessary prerequisite
            for simulation in EnergyPlus. (Default: False).
        print_progress: Set to True to have the progress of the translation
            printed as it is completed. (Default: False).

    Usage:

    .. code-block:: python

        import os
        from honeybee.model import Model
        from honeybee.room import Room
        from honeybee.config import folders
        from honeybee_energy.lib.programtypes import office_program
        import openstudio
        from honeybee_openstudio.writer import model_to_openstudio

        # Crate an input Model
        room = Room.from_box('Tiny_House_Zone', 5, 10, 3)
        room.properties.energy.program_type = office_program
        room.properties.energy.add_default_ideal_air()
        hb_model = Model('Tiny_House', [room])

        # translate the honeybee model to an openstudio model
        os_model = model_to_openstudio(hb_model)

        # save the OpenStudio model to an OSM
        osm = os.path.join(folders.default_simulation_folder, 'in.osm')
        os_model.save(osm, overwrite=True)

        # save the OpenStudio model to an IDF file
        idf_translator = openstudio.energyplus.ForwardTranslator()
        workspace = idf_translator.translateModel(os_model)
        idf = os.path.join(folders.default_simulation_folder, 'in.idf')
        workspace.save(idf, overwrite=True)
    """
    # check the model and check for rooms if this is enforced
    assert isinstance(model, Model), \
        'Expected Honeybee Model for model_to_openstudio. Got {}.'.format(type(model))
    if enforce_rooms:
        assert len(model.rooms) != 0, \
            'Model contains no Rooms and therefore cannot be simulated in EnergyPlus.'

    # duplicate model to avoid mutating it as we edit it for energy simulation
    original_model = model
    model = model.duplicate()

    # scale the model if the units are not meters
    if model.units != 'Meters':
        model.convert_to_units('Meters')
    # remove degenerate geometry within native E+ tolerance of 0.01 meters
    try:
        model.remove_degenerate_geometry(0.01)
    except ValueError:
        error = 'Failed to remove degenerate Rooms.\nYour Model units system is: {}. ' \
            'Is this correct?'.format(original_model.units)
        raise ValueError(error)
    if triangulate_non_planar_orphaned:
        model.triangulate_non_planar_quads(0.01)

    # remove the HVAC from any Rooms lacking setpoints
    rem_msgs = model.properties.energy.remove_hvac_from_no_setpoints()
    if len(rem_msgs) != 0:
        print('\n'.join(rem_msgs))

    # auto-assign stories if there are none since most OpenStudio measures need these
    if len(model.stories) == 0 and len(model.rooms) != 0:
        model.assign_stories_by_floor_height()

    # reset the IDs to be derived from the display_names if requested
    if use_geometry_names:
        id_map = model.reset_ids()
        model.properties.energy.sync_detailed_hvac_ids(id_map['rooms'])
    if use_resource_names:
        model.properties.energy.reset_resource_ids()

    # resolve the properties across zones
    single_zones, zone_dict = model.properties.energy.resolve_zones()

    # make note of how the airflow will be modeled across the building
    vent_sim_control = model.properties.energy.ventilation_simulation_control
    use_simple_vent = True if vent_sim_control.vent_control_type == 'SingleZone' \
        or sys.version_info < (3, 0) else False  # AFN not supported in .NET

    # create the OpenStudio model object and set properties for speed
    os_model = OSModel() if seed_model is None else seed_model
    os_model.setStrictnessLevel(openstudio.StrictnessLevel('None'))
    os_model.setFastNaming(True)

    # setup the Building
    os_building = os_model.getBuilding()
    if model._display_name is not None:
        os_building.setName(clean_ep_string(model.display_name))
    else:
        os_building.setName(model.identifier)
    os_model.setDayofWeekforStartDay('Sunday')  # this avoids lots of warnings
    os_model.alwaysOnDiscreteSchedule()
    os_model.alwaysOffDiscreteSchedule()
    os_model.alwaysOnContinuousSchedule()
    if print_progress:
        print('Model prepared for translation')

    # write all of the schedules and type limits
    schedules, type_limits = [], []
    always_on_included = False
    all_scheds = model.properties.energy.schedules + \
        model.properties.energy.orphaned_trans_schedules
    for sched in all_scheds:
        if sched.identifier == 'Always On':
            always_on_included = True
        schedules.append(sched)
        t_lim = sched.schedule_type_limit
        if t_lim is not None:
            for val in type_limits:
                if val is t_lim:
                    break
            else:
                type_limits.append(t_lim)
    if not always_on_included:
        always_schedule = model.properties.energy._always_on_schedule()
        schedules.append(always_schedule)
    for stl in type_limits:
        schedule_type_limits_to_openstudio(stl, os_model)
    for sch in schedules:
        schedule_to_openstudio(sch, os_model, schedule_directory)
    if print_progress:
        print('Translated {} Schedules'.format(len(schedules)))

    # write all of the materials, constructions, and construction sets
    w_cons = (WindowConstruction, WindowConstructionShade, WindowConstructionDynamic)
    materials, constructions, dynamic_cons = [], [], []
    all_constrs = model.properties.energy.constructions + \
        generic_construction_set.constructions_unique
    for constr in set(all_constrs):
        try:
            if use_simple_window_constructions and isinstance(constr, w_cons):
                if isinstance(constr, WindowConstruction):
                    new_con = constr.to_simple_construction()
                elif isinstance(constr, WindowConstructionShade):
                    new_con = constr.window_construction.to_simple_construction()
                elif isinstance(constr, WindowConstructionDynamic):
                    new_con = constr.constructions[0].to_simple_construction()
                materials.extend(new_con.materials)
                constructions.append(new_con)
            else:
                materials.extend(constr.materials)
                constructions.append(constr)
                if constr.has_frame:
                    materials.append(constr.frame)
                if constr.has_shade:
                    if constr.window_construction in all_constrs:
                        constructions.pop(-1)  # avoid duplicate specification
                    if constr.is_switchable_glazing:
                        materials.append(constr.switched_glass_material)
                elif constr.is_dynamic:
                    dynamic_cons.append(constr)
        except AttributeError:
            try:  # AirBoundaryConstruction or ShadeConstruction
                constructions.append(constr)  # AirBoundaryConstruction
            except TypeError:
                pass  # ShadeConstruction; no need to write it
    for mat in set(materials):
        material_to_openstudio(mat, os_model)
    if print_progress:
        print('Translated {} Materials'.format(len(materials)))
    for constr in constructions:
        construction_to_openstudio(constr, os_model)
    if print_progress:
        print('Translated {} Constructions'.format(len(constructions)))
    os_generic_c_set = construction_set_to_openstudio(generic_construction_set, os_model)
    os_building.setDefaultConstructionSet(os_generic_c_set)
    c_sets = model.properties.energy.construction_sets
    for con_set in c_sets:
        construction_set_to_openstudio(con_set, os_model)
    if print_progress:
        print('Translated {} Construction Sets'.format(len(c_sets)))

    # translate all of the programs
    p_types = model.properties.energy.program_types
    for program in p_types:
        program_type_to_openstudio(program, os_model, use_simple_vent)
    if print_progress:
        print('Translated {} Program Types'.format(len(p_types)))

    # create all of the spaces with all of their geometry
    if print_progress:
        print('Translating Rooms')
    space_map, story_map = {}, {}
    adj_map = {'faces': {}, 'sub_faces': {}}
    rooms = model.rooms
    for i, room in enumerate(rooms):
        os_space = room_to_openstudio(room, os_model, adj_map, use_simple_vent,
                                      triangulate_subfaces)
        space_map[room.identifier] = os_space
        try:
            story_map[room.story].append(os_space)
        except KeyError:  # first room found on the story
            story_map[room.story] = [os_space]
        if print_progress and (i + 1) % 100 == 0:
            print('  Translated {} out of {} Rooms'.format(i + 1, len(rooms)))
    if print_progress:
        print('Translated all {} Rooms'.format(len(rooms)))

    # create all of the zones
    if print_progress:
        print('Translating Zones')
    zone_map, zone_count = {}, 0
    for room in single_zones:
        os_zone = OSThermalZone(os_model)
        os_zone.setName(room.identifier)
        if room._display_name is not None:
            os_zone.setDisplayName(room.display_name)
        os_space = space_map[room.identifier]
        os_space.setThermalZone(os_zone)
        zone_map[room.identifier] = os_zone
        if room.multiplier != 1:
            os_zone.setMultiplier(room.multiplier)
        os_zone.setCeilingHeight(room.geometry.max.z - room.geometry.min.z)
        os_zone.setVolume(room.volume)
        if room.properties.energy.setpoint is not None:
            set_pt = room.properties.energy.setpoint
            therm = setpoint_to_openstudio_thermostat(set_pt, os_model, room.identifier)
            os_zone.setThermostatSetpointDualSetpoint(therm)
            humid = setpoint_to_openstudio_humidistat(set_pt, os_model, room.identifier)
            if humid is not None:
                os_zone.setZoneControlHumidistat(humid)
        daylight = room.properties.energy.daylighting_control
        if daylight is not None:
            dl_name = '{}_Daylighting'.format(room.identifier)
            os_daylight = os_model.getDaylightingControlByName(dl_name)
            if os_daylight.is_initialized():
                os_daylight = os_daylight.get()
                os_zone.setPrimaryDaylightingControl(os_daylight)
                os_zone.setFractionofZoneControlledbyPrimaryDaylightingControl(
                    daylight.control_fraction)
        zone_count += 1
        if print_progress and zone_count % 100 == 0:
            print('  Translated {} Zones'.format(zone_count))
    for zone_id, zone_data in zone_dict.items():
        rooms, z_prop, set_pt, vent = zone_data
        mult, ceil_hgt, vol, _, _ = z_prop
        os_zone = OSThermalZone(os_model)
        os_zone.setName(zone_id)
        for room in rooms:
            os_space = space_map[room.identifier]
            os_space.setThermalZone(os_zone)
            zone_map[room.identifier] = os_zone
        if mult != 1:
            os_zone.setMultiplier(mult)
        os_zone.setCeilingHeight(ceil_hgt)
        os_zone.setVolume(vol)
        if set_pt is not None:
            therm = setpoint_to_openstudio_thermostat(set_pt, os_model, zone_id)
            os_zone.setThermostatSetpointDualSetpoint(therm)
            humid = setpoint_to_openstudio_humidistat(set_pt, os_model, zone_id)
            if humid is not None:
                os_zone.setZoneControlHumidistat(humid)
        zone_count += 1
        if print_progress and zone_count % 100 == 0:
            print('  Translated {} Zones'.format(zone_count))
    if print_progress:
        print('Translated all {} Zones'.format(zone_count))

    # triangulate any apertures or doors with more than 4 vertices
    tri_sub_faces = []
    if triangulate_subfaces:
        tri_apertures, _ = model.triangulated_apertures()
        for tri_aps in tri_apertures:
            for i, ap in enumerate(tri_aps):
                if i != 0:
                    ap.properties.energy.vent_opening = None
                os_ap = aperture_to_openstudio(ap, os_model)
                os_face = adj_map['faces'][ap.parent.identifier]
                os_ap.setSurface(os_face)
                adj_map['sub_faces'][ap.identifier] = os_ap
                tri_sub_faces.append(ap)
        tri_doors, _ = model.triangulated_doors()
        for tri_drs in tri_doors:
            for i, dr in enumerate(tri_drs):
                if i != 0:
                    dr.properties.energy.vent_opening = None
                os_dr = door_to_openstudio(dr, os_model)
                os_face = adj_map['faces'][dr.parent.identifier]
                os_dr.setSurface(os_face)
                adj_map['sub_faces'][dr.identifier] = os_dr
                tri_sub_faces.append(dr)

    # assign stories to the rooms
    for story_id, os_spaces in story_map.items():
        story = OSBuildingStory(os_model)
        if story_id is not None:  # the users has specified the name of the story
            story.setName(story_id)
        else:  # give the room a dummy story so that it works with David's measures
            story.setName('UndefinedStory')
        for os_space in os_spaces:
            os_space.setBuildingStory(story)

    # assign adjacencies to all of the rooms
    already_adj = set()
    for room in model.rooms:
        for face in room.faces:
            if isinstance(face.boundary_condition, Surface):
                if face.identifier not in already_adj:
                    # add the adjacency to the set
                    adj_id = face.boundary_condition.boundary_condition_object
                    already_adj.add(adj_id)
                    # get the openstudio Surfaces and set the adjacency
                    try:
                        base_os_face = adj_map['faces'][face.identifier]
                        adj_os_face = adj_map['faces'][adj_id]
                    except KeyError:
                        msg = 'Missing adjacency exists between Face "{}" ' \
                            'and Face "{}."'.format(face.identifier, adj_id)
                        print(msg)
                    base_os_face.setAdjacentSurface(adj_os_face)
                    # set the adjacency of all sub-faces
                    for sub_face in face.sub_faces:
                        if len(sub_face.geometry) <= 4 or not triangulate_subfaces:
                            adj_id = sub_face.boundary_condition.boundary_condition_object
                            try:
                                os_sub_face = adj_map['sub_faces'][sub_face.identifier]
                                adj_os_sub_face = adj_map['sub_faces'][adj_id]
                            except KeyError:
                                msg = 'Missing adjacency exists between subface "{}" ' \
                                    'and subface "{}."'.format(
                                        sub_face.identifier, adj_id)
                                print(msg)
                            os_sub_face.setAdjacentSubSurface(adj_os_sub_face)
    for sub_face in tri_sub_faces:
        if isinstance(sub_face.boundary_condition, Surface):
            adj_id = sub_face.boundary_condition.boundary_condition_object
            try:
                os_sub_face = adj_map['sub_faces'][sub_face.identifier]
                adj_os_sub_face = adj_map['sub_faces'][adj_id]
            except KeyError:
                msg = 'Missing adjacency exists between subface "{}" ' \
                    'and subface "{}."'.format(sub_face.identifier, adj_id)
                print(msg)
            os_sub_face.setAdjacentSubSurface(adj_os_sub_face)

    # if simple ventilation is being used, write the relevant objects
    if use_simple_vent:
        for room in model.rooms:  # add simple add air mixing and window ventilation
            for face in room.faces:
                if isinstance(face.type, AirBoundary):  # write the air mixing objects
                    try:
                        adj_room = face.boundary_condition.boundary_condition_objects[-1]
                        target_zone = zone_map[room.identifier]
                        source_zone = zone_map[adj_room]
                        air_mixing_to_openstudio(face, target_zone, source_zone, os_model)
                    except AttributeError as e:
                        raise ValueError(
                            'Face "{}" is an Air Boundary but lacks a Surface boundary '
                            'condition.\n{}'.format(face.full_id, e))
                # add simple window ventilation objects where applicable
                if isinstance(face.boundary_condition, Outdoors):
                    for sub_f in face.sub_faces:
                        vent_open = sub_f.properties.energy.vent_opening
                        if vent_open is not None:
                            os_vent = ventilation_opening_to_openstudio(vent_open, os_model)
                            os_vent.addToThermalZone(zone_map[room.identifier])
    else:  # we are using the AFN!
        # create the AFN reference crack for the model and make an outdoor sensor
        vent_sim_ctrl = model.properties.energy.ventilation_simulation_control
        os_ref_crack = ventilation_sim_control_to_openstudio(vent_sim_ctrl, os_model)
        prog_manager = ventilation_control_program_manager(os_model)  # EMS manager
        outdoor_temperature_sensor(os_model)  # add an EMS sensor for outdoor temperature
        #  loop though the geometry and assign all AFN properties
        zone_air_nodes = {}  # track EMS zone air temperature sensors
        set_by_adj = set()  # track the objects with properties set by adjacency
        for room in model.rooms:  # write an AirflowNetworkZone object in for the Room
            os_zone = zone_map[room.identifier]
            os_afn_room_node = os_zone.getAirflowNetworkZone()
            os_afn_room_node.setVentilationControlMode('NoVent')
            operable_sub_fs = []  # collect the sub-face objects for the EMS
            opening_factors = []  # collect the maximum opening factors for the EMS
            for face in room.faces:  # write AFN crack infiltration for the Face
                if face.identifier not in set_by_adj:
                    vent_crack = face.properties.energy.vent_crack
                    if vent_crack is not None:
                        os_crack = afn_crack_to_openstudio(
                            vent_crack, os_model, os_ref_crack)
                        os_crack.setName('{}_Crack'.format(face.identifier))
                        os_face = adj_map['faces'][face.identifier]
                        os_face.getAirflowNetworkSurface(os_crack)
                        if isinstance(face.boundary_condition, Surface):
                            adj_id = face.boundary_condition.boundary_condition_object
                            set_by_adj.add(adj_id)
                    for sub_f in face.sub_faces:  # write AFN openings for each sub-face
                        vent_open = sub_f.properties.energy.vent_opening
                        if vent_open is not None:
                            os_opening, op_fac = ventilation_opening_to_openstudio_afn(
                                vent_open, os_model, os_ref_crack)
                            os_sub_f = adj_map['sub_faces'][sub_f.identifier]
                            os_afn_sf = os_sub_f.getAirflowNetworkSurface(os_opening)
                            if op_fac is not None:
                                operable_sub_fs.append(os_sub_f)
                                opening_factors.append(op_fac)
                                op_fac = 1 if op_fac == 0 else op_fac
                                os_afn_sf.setWindowDoorOpeningFactorOrCrackFactor(op_fac)
            # translate the Room's VentilationControl to an EMS program
            vent_control = room.properties.energy.window_vent_control
            if vent_control is not None:
                try:
                    zone_node = zone_air_nodes[os_zone.nameString()]
                except KeyError:
                    zone_node = zone_temperature_sensor(os_zone, os_model)
                    zone_air_nodes[os_zone.nameString()] = zone_node
                os_ems_program = ventilation_control_to_openstudio_afn(
                    vent_control, opening_factors, operable_sub_fs,
                    zone_node, os_model, room.identifier)
                prog_manager.addProgram(os_ems_program)

    if print_progress:
        print('Assigned adjacencies to all Rooms')
    if print_progress:
        print('Translating Systems')

    # write any ventilation fan definitions
    for room in model.rooms:
        for fan in room.properties.energy.fans:
            os_fan = ventilation_fan_to_openstudio(fan, os_model)
            os_fan.setName('{}..{}'.format(fan.identifier, room.identifier))
            os_fan.addToThermalZone(zone_map[room.identifier])

    # assign HVAC systems to all of the rooms
    zone_rooms = {room.zone: room for room in single_zones}
    for zone_id, zone_data in zone_dict.items():
        for room in zone_data[0]:
            if room.properties.energy.hvac is not None:
                zone_rooms[zone_id] = room
                break
    ideal_air_count = 0
    template_zones, template_hvac_dict, detailed_hvac_dict = {}, {}, {}
    for zone_id, room in zone_rooms.items():
        hvac = room.properties.energy.hvac
        os_zone = zone_map[room.identifier]
        if isinstance(hvac, IdealAirSystem):
            os_hvac = ideal_air_system_to_openstudio(hvac, os_model, room)
            if room.identifier != zone_id:
                os_hvac.setName('{} Ideal Loads Air System'.format(zone_id))
            os_hvac.addToThermalZone(os_zone)
            ideal_air_count += 1
        elif isinstance(hvac, _TemplateSystem):
            template_hvac_dict[hvac.identifier] = hvac
            set_pt = room.properties.energy.setpoint
            if set_pt is not None:
                try:
                    zone_list = template_zones[hvac.identifier]
                except KeyError:  # first zone found in the HVAC
                    zone_list = {'heated_zones': [], 'cooled_zones': []}
                    template_zones[hvac.identifier] = zone_list
                if set_pt.heating_setpoint > 5:
                    zone_list['heated_zones'].append(os_zone)
                if set_pt.cooling_setpoint < 33:
                    zone_list['cooled_zones'].append(os_zone)
        elif isinstance(hvac, DetailedHVAC):
            detailed_hvac_dict[hvac.identifier] = hvac
    if print_progress and ideal_air_count != 0:
        print('  Assigned {} Ideal Air Systems'.format(ideal_air_count))
    # translate template HVAC systems
    os_model.setFastNaming(False)
    if len(template_hvac_dict) != 0:
        for hvac_id, os_zones in template_zones.items():
            hvac = template_hvac_dict[hvac_id]
            template_hvac_to_openstudio(hvac, os_zones, os_model)
            if print_progress:
                print('  Assigned template HVAC: {}'.format(hvac.display_name))
        if len(template_zones) != 0:  # rename air and plant loop nodes for readability
            rename_air_loop_nodes(os_model)
            rename_plant_loop_nodes(os_model)
    # translate detailed HVAC systems
    if len(detailed_hvac_dict) != 0:
        assert hbe_folders.ironbug_exe is not None, 'Detailed Ironbug HVAC System was ' \
            'assigned but no Ironbug installation was found.'
        for hvac_id, hvac in detailed_hvac_dict.items():
            hvac_trans_dir = tempfile.gettempdir()
            spec_file = os.path.join(hvac_trans_dir, '{}.json'.format(hvac.identifier))
            with open(spec_file, 'w') as sf:
                json.dump(hvac.specification, sf)
            osm_file = os.path.join(hvac_trans_dir, '{}.osm'.format(hvac.identifier))
            os_model.save(os_path(osm_file), overwrite=True)
            cmds = [hbe_folders.ironbug_exe, osm_file, spec_file]
            process = subprocess.Popen(
                cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            result = process.communicate()  # pause script until command is done
            exist_os_model = OSModel.load(os_path(osm_file))
            if exist_os_model.is_initialized():
                os_model = exist_os_model.get()
            else:
                msg = 'Failed to apply Detailed HVAC "{}"\n{}\n{}'.format(
                    hvac_id, result[0], result[1])
                raise ValueError(msg)
            if print_progress:
                print('  Assigned detailed HVAC: {}'.format(hvac.display_name))

    # write service hot water and any SHW systems
    shw_sys_dict = {}
    for room in model.rooms:
        hot_water = room.properties.energy.service_hot_water
        if hot_water is not None and hot_water.flow_per_area != 0:
            os_shw_conn = hot_water_to_openstudio(hot_water, room, os_model)
            total_flow = (hot_water.flow_per_area / 3600000.) * room.floor_area
            water_temp = hot_water.target_temperature
            shw_sys = room.properties.energy.shw
            shw_sys_id = shw_sys.identifier \
                if shw_sys is not None else 'Default_District_SHW'
            try:  # try to add the hot water to the existing system
                shw_sys_props = shw_sys_dict[shw_sys_id]
                shw_sys_props[1].append(os_shw_conn)
                shw_sys_props[2] += total_flow
                if water_temp > shw_sys_props[3]:
                    shw_sys_props[3] = water_temp
            except KeyError:  # first time that the SHW system is encountered
                shw_sys_props = [shw_sys, [os_shw_conn], total_flow, water_temp]
                shw_sys_dict[shw_sys_id] = shw_sys_props
    if len(shw_sys_dict) != 0:
        # add all of the SHW Systems to the model
        for shw_sys_props in shw_sys_dict.values():
            shw_sys, os_shw_conns, total_flow, w_temp = shw_sys_props
            shw_system_to_openstudio(shw_sys, os_shw_conns, total_flow, w_temp, os_model)
            if print_progress:
                shw_sys_name = shw_sys.display_name \
                    if shw_sys is not None else 'Default_District_SHW'
                print('  Assigned SHW System: {}'.format(shw_sys_name))

    # write any EMS programs for dynamic constructions
    if len(dynamic_cons) != 0:
        # create the program calling manager
        os_prog_manager = OSEnergyManagementSystemProgramCallingManager(os_model)
        os_prog_manager.setName('Dynamic_Window_Constructions')
        os_prog_manager.setCallingPoint('BeginTimestepBeforePredictor')
        # get all of the sub-faces with the dynamic construction
        dyn_dict = {}
        for room in model.rooms:
            for face in room.faces:
                for sf in face.sub_faces:
                    con = sf.properties.energy.construction
                    if isinstance(con, WindowConstructionDynamic):
                        os_sf = adj_map['sub_faces'][sf.identifier]
                        try:
                            dyn_dict[con.identifier].append(os_sf)
                        except KeyError:
                            dyn_dict[con.identifier] = [os_sf]
        for con in dynamic_cons:
            ems_program = window_dynamic_ems_program_to_openstudio(
                con, dyn_dict[con.identifier], os_model)
            os_prog_manager.addProgram(ems_program)

    # add the orphaned objects
    shade_count, shades_to_group = 0, []
    for face in model.orphaned_faces:
        shades_to_group.append(face_to_openstudio(face, os_model))
        shade_count += 1
    for aperture in model.orphaned_apertures:
        shades_to_group.append(aperture_to_openstudio(aperture, os_model))
        shade_count += 1
    for door in model.orphaned_doors:
        shades_to_group.append(door_to_openstudio(door, os_model))
        shade_count += 1
    for shade in model.orphaned_shades:
        shades_to_group.append(shade_to_openstudio(shade, os_model))
        shade_count += 1
    for shade_mesh in model.shade_meshes:
        shade_mesh_to_openstudio(shade_mesh, os_model)
        shade_count += 1
    if len(shades_to_group) != 0:
        shd_group = OSShadingSurfaceGroup(os_model)
        shd_group.setName('Orphaned Shades')
        shd_group.setShadingSurfaceType('Building')
        for os_shade in shades_to_group:
            os_shade.setShadingSurfaceGroup(shd_group)
    if print_progress and shade_count != 0:
        print('Translated {} Shades'.format(shade_count))

    # write the electric load center is any generator objects are in the model
    os_pv_gens = os_model.getGeneratorPVWattss()
    if os_vector_len(os_pv_gens) != 0:
        load_center = model.properties.energy.electric_load_center
        electric_load_center_to_openstudio(load_center, os_pv_gens, os_model)

    # return the Model object
    return os_model


def model_to_osm(
    model, seed_model=None, schedule_directory=None,
    use_geometry_names=False, use_resource_names=False, print_progress=False
):
    """Translate a Honeybee Model to an OSM string.

    Args:
        model: The Honeybee Model to be converted into an OpenStudio Model.
        seed_model: An optional OpenStudio Model object to which the Honeybee
            Model will be added. If None, a new OpenStudio Model will be
            initialized within this method. (Default: None).
        schedule_directory: An optional file directory to which all file-based
            schedules should be written to. If None, all ScheduleFixedIntervals
            will be translated to Schedule:Compact and written fully into the
            IDF string instead of to Schedule:File. (Default: None).
        use_geometry_names: Boolean to note whether a cleaned version of all
            geometry display names should be used instead of identifiers when
            translating the Model to OSM and IDF. Using this flag will affect
            all Rooms, Faces, Apertures, Doors, and Shades. It will generally
            result in more read-able names in the OSM and IDF but this means
            that it will not be easy to map the EnergyPlus results back to the
            input Honeybee Model. Cases of duplicate IDs resulting from
            non-unique names will be resolved by adding integers to the ends
            of the new IDs that are derived from the name. (Default: False).
        use_resource_names: Boolean to note whether a cleaned version of all
            resource display names should be used instead of identifiers when
            translating the Model to OSM and IDF. Using this flag will affect
            all Materials, Constructions, ConstructionSets, Schedules, Loads,
            and ProgramTypes. It will generally result in more read-able names
            for the resources in the OSM and IDF. Cases of duplicate IDs
            resulting from non-unique names will be resolved by adding integers
            to the ends of the new IDs that are derived from the name. (Default: False).
        print_progress: Set to True to have the progress of the translation
            printed as it is completed. (Default: False).
    """
    # check that the input is a model
    assert isinstance(model, Model), \
        'Expected Honeybee Model for model_to_osm. Got {}.'.format(type(model))
    # translate the Honeybee Model to an OpenStudio Model
    os_model = model_to_openstudio(
        model, seed_model, schedule_directory, use_geometry_names, use_resource_names,
        print_progress=print_progress
    )
    return str(os_model)


def model_to_idf(
    model, seed_model=None, schedule_directory=None,
    use_geometry_names=False, use_resource_names=False, print_progress=False
):
    """Translate a Honeybee Model to an IDF string using OpenStudio SDK translators.

    Args:
        model: The Honeybee Model to be converted into an OpenStudio Model.
        seed_model: An optional OpenStudio Model object to which the Honeybee
            Model will be added. If None, a new OpenStudio Model will be
            initialized within this method. (Default: None).
        schedule_directory: An optional file directory to which all file-based
            schedules should be written to. If None, all ScheduleFixedIntervals
            will be translated to Schedule:Compact and written fully into the
            IDF string instead of to Schedule:File. (Default: None).
        use_geometry_names: Boolean to note whether a cleaned version of all
            geometry display names should be used instead of identifiers when
            translating the Model to OSM and IDF. Using this flag will affect
            all Rooms, Faces, Apertures, Doors, and Shades. It will generally
            result in more read-able names in the OSM and IDF but this means
            that it will not be easy to map the EnergyPlus results back to the
            input Honeybee Model. Cases of duplicate IDs resulting from
            non-unique names will be resolved by adding integers to the ends
            of the new IDs that are derived from the name. (Default: False).
        use_resource_names: Boolean to note whether a cleaned version of all
            resource display names should be used instead of identifiers when
            translating the Model to OSM and IDF. Using this flag will affect
            all Materials, Constructions, ConstructionSets, Schedules, Loads,
            and ProgramTypes. It will generally result in more read-able names
            for the resources in the OSM and IDF. Cases of duplicate IDs
            resulting from non-unique names will be resolved by adding integers
            to the ends of the new IDs that are derived from the name. (Default: False).
        print_progress: Set to True to have the progress of the translation
            printed as it is completed. (Default: False).
    """
    # check that the input is a model
    assert isinstance(model, Model), \
        'Expected Honeybee Model for model_to_idf. Got {}.'.format(type(model))
    # translate the Honeybee Model to an OpenStudio Model
    os_model = model_to_openstudio(
        model, seed_model, schedule_directory, use_geometry_names, use_resource_names,
        print_progress=print_progress
    )
    # translate the model to an IDF string
    if (sys.version_info < (3, 0)):
        idf_translator = openstudio.EnergyPlusForwardTranslator()
    else:
        idf_translator = openstudio.energyplus.ForwardTranslator()
    workspace = idf_translator.translateModel(os_model)
    return str(workspace)


def model_to_gbxml(
    model, triangulate_non_planar_orphaned=True, triangulate_subfaces=False,
    full_geometry=False, interior_face_type=None, ground_face_type=None,
    print_progress=False
):
    """Translate a Honeybee Model to gbXML string using OpenStudio SDK translators.

    Args:
        model: The Honeybee Model to be converted into an OpenStudio Model.
        triangulate_non_planar_orphaned: Boolean to note whether any non-planar
            orphaned geometry in the model should be triangulated.
            This can be helpful because OpenStudio simply raises an error when
            it encounters non-planar geometry, which would hinder the ability
            to save files that are to be corrected later. (Default: False).
        triangulate_subfaces: Boolean to note whether sub-faces (including
            Apertures and Doors) should be triangulated if they have more
            than 4 sides (True) or whether they should be left as they are (False).
            This triangulation is necessary when exporting directly to EnergyPlus
            since it cannot accept sub-faces with more than 4 vertices. (Default: True).
        full_geometry: Boolean to note whether space boundaries and shell geometry
            should be included in the exported gbXML vs. just the minimal required
            non-manifold geometry. (Default: False).
        interior_face_type: Text string for the type to be used for all interior
            floor faces. If unspecified, the interior types will be left as they are.
            Choose from the following. InteriorFloor, Ceiling.
        ground_face_type: Text string for the type to be used for all ground-contact
            floor faces. If unspecified, the ground types will be left as they are.
            Choose from the following. UndergroundSlab, SlabOnGrade, RaisedFloor.
        print_progress: Set to True to have the progress of the translation
            printed as it is completed. (Default: False).
    """
    # check that the input is a model
    assert isinstance(model, Model), \
        'Expected Honeybee Model for model_to_gbxml. Got {}.'.format(type(model))

    # remove degenerate geometry within native DesignBuilder tolerance of 0.02 meters
    original_model = model
    model = model.duplicate()  # duplicate to avoid mutating the input
    if model.units != 'Meters':
        model.convert_to_units('Meters')
    try:
        model.remove_degenerate_geometry(0.02)
    except ValueError:
        error = 'Failed to remove degenerate Rooms.\nYour Model units system is: {}. ' \
            'Is this correct?'.format(original_model.units)
        raise ValueError(error)

    # remove any detailed HVAC or AFN as this will only slow the translation down
    v_control = model.properties.energy.ventilation_simulation_control
    det_hvac_count = 0
    for hvac in model.properties.energy.hvacs:
        if hvac is not None and not isinstance(hvac, IdealAirSystem):
            det_hvac_count += 1
    if v_control.vent_control_type != 'SingleZone' or det_hvac_count != 0:
        for room in model.rooms:
            room.properties.energy.assign_ideal_air_equivalent()
        v_control.vent_control_type = 'SingleZone'

    # translate the Honeybee Model to an OpenStudio Model
    os_model = model_to_openstudio(
        model, triangulate_non_planar_orphaned=triangulate_non_planar_orphaned,
        triangulate_subfaces=triangulate_subfaces,
        use_simple_window_constructions=True, print_progress=print_progress
    )

    # translate the model to a gbXML string
    if (sys.version_info < (3, 0)):
        gbxml_translator = openstudio.GbXMLForwardTranslator()
    else:
        gbxml_translator = openstudio.gbxml.GbXMLForwardTranslator()
    gbxml_str = gbxml_translator.modelToGbXMLString(os_model)

    # replace all interior floors with the specified type
    if interior_face_type == 'InteriorFloor':
        gbxml_str = gbxml_str.replace('="Ceiling"', '="InteriorFloor"')
    elif interior_face_type == 'Ceiling':
        gbxml_str = gbxml_str.replace('="InteriorFloor"', '="Ceiling"')

    # replace all ground floors with the specified type
    if ground_face_type == 'UndergroundSlab':
        gbxml_str = gbxml_str.replace('="SlabOnGrade"', '="UndergroundSlab"')
        gbxml_str = gbxml_str.replace('="RaisedFloor"', '="UndergroundSlab"')
    elif ground_face_type == 'SlabOnGrade':
        gbxml_str = gbxml_str.replace('="UndergroundSlab"', '="SlabOnGrade"')
        gbxml_str = gbxml_str.replace('="RaisedFloor"', '="SlabOnGrade"')
    elif ground_face_type == 'RaisedFloor':
        gbxml_str = gbxml_str.replace('="UndergroundSlab"', '="RaisedFloor"')
        gbxml_str = gbxml_str.replace('="SlabOnGrade"', '="RaisedFloor"')

    # write the SpaceBoundary and ShellGeometry into the XML if requested
    if full_geometry:
        # get a dictionary of rooms in the model
        room_dict = {room.identifier: room for room in model.rooms}

        # register all of the namespaces within the OpenStudio-exported XML
        ET.register_namespace('', 'http://www.gbxml.org/schema')
        ET.register_namespace('xhtml', 'http://www.w3.org/1999/xhtml')
        ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        ET.register_namespace('xsd', 'http://www.w3.org/2001/XMLSchema')

        # parse the XML and get the building definition
        root = ET.fromstring(gbxml_str)
        gbxml_header = r'{http://www.gbxml.org/schema}'
        building = root[0][1]

        # loop through surfaces in the gbXML so that we know the name of the interior ones
        surface_set = set()
        for room_element in root[0].findall(gbxml_header + 'Surface'):
            surface_set.add(room_element.get('id'))

        # loop through the rooms in the XML and add them as space boundaries to the room
        for room_element in building.findall(gbxml_header + 'Space'):
            room_id = room_element.get('zoneIdRef')
            if room_id:
                room_id = room_element.get('id')
                shell_element = ET.Element('ShellGeometry')
                shell_element.set('id', '{}Shell'.format(room_id))
                shell_geo_element = ET.SubElement(shell_element, 'ClosedShell')
                hb_room = room_dict[room_id[:-6]]  # remove '_Space' from the end
                for face in hb_room:
                    face_xml, face_geo_xml = _face_to_gbxml_geo(face, surface_set)
                    if face_xml is not None:
                        room_element.append(face_xml)
                        shell_geo_element.append(face_geo_xml)
                room_element.append(shell_element)

        # convert the element tree back into a string
        if sys.version_info >= (3, 0):
            gbxml_str = ET.tostring(root, encoding='unicode', xml_declaration=True)
        else:
            gbxml_str = ET.tostring(root)

    return gbxml_str


def _face_to_gbxml_geo(face, face_set):
    """Get an Element Tree of a gbXML SpaceBoundary for a Face.

    Note that the resulting string is only meant to go under the "Space" tag and
    it is not a Surface tag with all of the construction and boundary condition
    properties assigned to it.

    Args:
        face: A honeybee Face for which an gbXML representation will be returned.
        face_set: A set of surface identifiers in the model, used to evaluate whether
            the geometry must be associated with its boundary condition surface.

    Returns:
        A tuple with two elements.

        -   face_element: The element tree for the SpaceBoundary definition of the Face.

        -   loop_element: The element tree for the PolyLoop definition of the Face,
            which is useful in defining the shell.
    """
    # create the face element and associate it with a surface in the model
    face_element = ET.Element('SpaceBoundary')
    face_element.set('isSecondLevelBoundary', 'false')
    obj_id = None
    if face.identifier in face_set:
        obj_id = face.identifier
    elif isinstance(face.boundary_condition, Surface):
        bc_obj = face.boundary_condition.boundary_condition_object
        if bc_obj in face_set:
            obj_id = bc_obj
    if obj_id is None:
        return None, None
    face_element.set('surfaceIdRef', obj_id)

    # write the geometry of the face
    geo_element = ET.SubElement(face_element, 'PlanarGeometry')
    loop_element = ET.SubElement(geo_element, 'PolyLoop')
    for pt in face.vertices:
        pt_element = ET.SubElement(loop_element, 'CartesianPoint')
        for coord in pt:
            coord_element = ET.SubElement(pt_element, 'Coordinate')
            coord_element.text = str(coord)
    return face_element, loop_element
