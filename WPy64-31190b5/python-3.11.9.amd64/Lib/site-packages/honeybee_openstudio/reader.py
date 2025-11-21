# coding=utf-8
"""Methods to read OpenStudio Models into Honeybee Models."""
from __future__ import division
import sys
import os

from ladybug_geometry.geometry3d import Point3D, Face3D
from honeybee.typing import clean_string, clean_ep_string
from honeybee.altnumber import autocalculate
from honeybee.boundarycondition import Outdoors, Surface, boundary_conditions
from honeybee.facetype import face_types
from honeybee.shade import Shade
from honeybee.door import Door
from honeybee.aperture import Aperture
from honeybee.face import Face
from honeybee.room import Room
from honeybee.model import Model
from honeybee_energy.boundarycondition import Adiabatic, OtherSideTemperature
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.construction.windowshade import WindowConstructionShade
from honeybee_energy.construction.dynamic import WindowConstructionDynamic

from honeybee_openstudio.openstudio import openstudio, os_vector_len, os_path
from honeybee_openstudio.schedule import extract_all_schedules
from honeybee_openstudio.construction import extract_all_constructions, \
    shade_construction_from_openstudio
from honeybee_openstudio.constructionset import construction_set_from_openstudio
from honeybee_openstudio.load import people_from_openstudio, lighting_from_openstudio, \
    electric_equipment_from_openstudio, gas_equipment_from_openstudio, \
    process_from_openstudio, hot_water_from_openstudio, infiltration_from_openstudio, \
    ventilation_from_openstudio, setpoint_from_openstudio_thermostat, \
    setpoint_from_openstudio_humidistat, daylight_from_openstudio
from honeybee_openstudio.programtype import program_type_from_openstudio
from honeybee_openstudio.hvac.idealair import ideal_air_system_from_openstudio

NATIVE_EP_TOL = 0.01  # native tolerance of E+ in meters
GLASS_CONSTR = (WindowConstruction, WindowConstructionShade, WindowConstructionDynamic)


def face_3d_from_openstudio(os_vertices):
    """Convert an OpenStudio Point3dVector into a Face3D.

    Args:
        os_vertices: An OpenStudio Point3dVector that came from any OpenStudio
            PlanarSurface class.

    Returns:
        A ladybug-geometry Face3D object created from the OpenStudio Point3dVector.
    """
    # create all of the Point3D objects
    vertices = []
    for v in os_vertices:
        vertices.append(Point3D(v.x(), v.y(), v.z()))
    face_3d = Face3D(vertices)

    # sense if the vertices loop back on themselves to cut holes
    separate_holes = False
    for i, pt in enumerate(vertices):
        if i + 2 >= len(vertices):
            break
        for j in range(i + 2, len(vertices)):
            if pt.is_equivalent(vertices[j], NATIVE_EP_TOL):
                separate_holes = True
                break

    # separate the boundary and holes if necessary
    if separate_holes:
        try:
            return face_3d.separate_boundary_and_holes(NATIVE_EP_TOL)
        except AssertionError:  # invalid face to be removed later
            pass
    return face_3d


def shades_from_openstudio(os_shade_group, constructions=None, schedules=None):
    """Convert an OpenStudio ShadingSurfaceGroup into a list of Honeybee Shades.

    Args:
        os_shade_group: An OpenStudio ShadingSurfaceGroup to be converted into
            a list of Honeybee Shades.
        constructions: An optional dictionary of Honeybee Construction objects
            which will be used to assign constructions to the shades.
        schedules: An optional dictionary of Honeybee Schedule objects which will
            be used to assign transmittance schedules to the shades.

    Returns:
        A list of Honeybee Shade objects.
    """
    # get variables that apply to the whole group
    shades = []
    os_site_transform = os_shade_group.siteTransformation()
    shade_type = os_shade_group.shadingSurfaceType()
    is_detached = False if shade_type == 'Space' else True

    # loop through the geometries and create the shade objects
    for os_shade in os_shade_group.shadingSurfaces():
        # create the shade object
        os_vertices = os_site_transform * os_shade.vertices()
        geo = face_3d_from_openstudio(os_vertices)
        shade = Shade(clean_string(os_shade.nameString()), geo, is_detached)
        if os_shade.displayName().is_initialized():
            shade.display_name = os_shade.displayName().get()

        # assign the construction if it exists
        if constructions is not None and not os_shade.isConstructionDefaulted():
            construction = os_shade.construction()
            if construction.is_initialized():
                const_name = \
                    '{} Shade'.format(clean_ep_string(construction.get().nameString()))
                try:
                    shade.properties.energy.construction = constructions[const_name]
                except KeyError:  # make a new shade construction
                    const_obj = construction.get()
                    const = const_obj.to_LayeredConstruction().get()
                    construction = shade_construction_from_openstudio(const)
                    shade.properties.energy.construction = construction
                    constructions[const_name] = const

        # assign the transmittance schedule if it exists
        trans_sch = os_shade.transmittanceSchedule()
        if schedules is not None and trans_sch.is_initialized():
            t_sch_name = clean_ep_string(trans_sch.get().nameString())
            if t_sch_name in schedules:
                try:
                    t_sched = schedules[t_sch_name]
                    shade.properties.energy.transmittance_schedule = t_sched
                except KeyError:
                    pass  # schedule was of a type that could not be loaded

        shades.append(shade)
    return shades


def _extract_sub_surface_bc(os_sub_surface):
    """Get a honeybee boundary condition from an OpenStudio SubSurface."""
    bc = None
    adjacent_sub_surface = os_sub_surface.adjacentSubSurface()
    if adjacent_sub_surface.is_initialized():
        adjacent_sub_surface = adjacent_sub_surface.get()
        adj_door_id = clean_string(adjacent_sub_surface.nameString())
        adj_face_id = None
        if adjacent_sub_surface.surface().is_initialized():
            adj_face_id = clean_string(adjacent_sub_surface.surface().get().nameString())
        adj_room_id = None
        if adjacent_sub_surface.space().is_initialized():
            adj_room_id = clean_string(adjacent_sub_surface.space().get().nameString())
        if adj_face_id is not None and adj_room_id is not None:
            bc = Surface((adj_door_id, adj_face_id, adj_room_id), sub_face=True)
    if bc is None:  # set it to outdoors
        if os_sub_surface.surface().is_initialized():
            os_surface = os_sub_surface.surface().get()
            sun_exposure = True if os_surface.sunExposure() == 'SunExposed' else False
            wind_exposure = True if os_surface.windExposure() == 'WindExposed' else False
            view_factor = os_sub_surface.viewFactortoGround()
            view_factor = view_factor.get() \
                if view_factor.is_initialized() else autocalculate
            bc = Outdoors(sun_exposure, wind_exposure, view_factor)
        else:
            bc = boundary_conditions.outdoors
    return bc


def door_from_openstudio(os_sub_surface, os_site_transform=None, constructions=None):
    """Convert an OpenStudio SubSurface into a Honeybee Door.

    Args:
        os_sub_surface: An OpenStudio SubSurface to be converted into a Honeybee Door.
        site_transform: An optional OpenStudio Transformation object that
            describes how the coordinates of the SubSurface object relate
            to the world coordinate system.
        constructions: An optional dictionary of Honeybee Construction objects
            which will be used to assign a construction to the door.

    Returns:
        A honeybee Door object.
    """
    # create the door object
    os_vertices = os_sub_surface.vertices()
    if os_site_transform is not None:
        os_vertices = os_site_transform * os_vertices
    geo = face_3d_from_openstudio(os_vertices)
    door = Door(clean_string(os_sub_surface.nameString()), geo)
    # assign the display name and type
    if os_sub_surface.displayName().is_initialized():
        door.display_name = os_sub_surface.displayName().get()
    if os_sub_surface.subSurfaceType() == 'GlassDoor':
        door.is_glass = True
    # assign the boundary condition
    door.boundary_condition = _extract_sub_surface_bc(os_sub_surface)
    # assign the construction if it exists
    if constructions is not None and not os_sub_surface.isConstructionDefaulted():
        construction = os_sub_surface.construction()
        if construction.is_initialized():
            const_name = clean_ep_string(construction.get().nameString())
            if const_name in constructions:
                con = constructions[const_name]
                if isinstance(con, GLASS_CONSTR):
                    door.is_glass = True
                door.properties.energy.construction = con
    return door


def aperture_from_openstudio(os_sub_surface, os_site_transform=None, constructions=None):
    """Convert an OpenStudio SubSurface into a Honeybee Aperture.

    Args:
        os_sub_surface: An OpenStudio SubSurface to be converted into
            a Honeybee Aperture.
        site_transform: An optional OpenStudio Transformation object that
            describes how the coordinates of the SubSurface object relate
            to the world coordinate system.
        constructions: An optional dictionary of Honeybee Construction objects
            which will be used to assign a construction to the aperture.

    Returns:
        A honeybee Aperture object.
    """
    # create the door object
    os_vertices = os_sub_surface.vertices()
    if os_site_transform is not None:
        os_vertices = os_site_transform * os_vertices
    geo = face_3d_from_openstudio(os_vertices)
    aperture = Aperture(clean_string(os_sub_surface.nameString()), geo)
    # assign the display name and type
    if os_sub_surface.displayName().is_initialized():
        aperture.display_name = os_sub_surface.displayName().get()
    if os_sub_surface.subSurfaceType() == 'OperableWindow':
        aperture.is_operable = True
    # assign the boundary condition
    aperture.boundary_condition = _extract_sub_surface_bc(os_sub_surface)
    # assign the construction if it exists
    if constructions is not None and not os_sub_surface.isConstructionDefaulted():
        construction = os_sub_surface.construction()
        if construction.is_initialized():
            const_name = clean_ep_string(construction.get().nameString())
            if const_name in constructions:
                con = constructions[const_name]
                if isinstance(con, GLASS_CONSTR):
                    aperture.properties.energy.construction = con
    return aperture


def face_from_openstudio(os_surface, os_site_transform=None, constructions=None):
    """Convert an OpenStudio Surface into a Honeybee Face.

    Args:
        os_surface: An OpenStudio Surface to be converted into a Honeybee Aperture.
        site_transform: An optional OpenStudio Transformation object that
            describes how the coordinates of the SubSurface object relate
            to the world coordinate system.
        constructions: An optional dictionary of Honeybee Construction objects
            which will be used to assign a construction to the aperture.

    Returns:
        A honeybee Aperture object.
    """
    # create the door object
    os_vertices = os_surface.vertices()
    if os_site_transform is not None:
        os_vertices = os_site_transform * os_vertices
    geo = face_3d_from_openstudio(os_vertices)
    face = Face(clean_string(os_surface.nameString()), geo)

    # assign the display name and type
    if os_surface.displayName().is_initialized():
        face.display_name = os_surface.displayName().get()
    face_type = os_surface.surfaceType()
    if os_surface.isAirWall():
        face.type = face_types.air_boundary
    elif 'Wall' in face_type:
        face.type = face_types.wall
    elif 'Floor' in face_type:
        face.type = face_types.floor
    elif 'Roof' in face_type or 'Ceiling' in face_type:
        face.type = face_types.roof_ceiling

    # assign the boundary condition
    bc = None
    surface_bc = os_surface.outsideBoundaryCondition()
    adjacent_surface = os_surface.adjacentSurface()
    if adjacent_surface.is_initialized():
        adjacent_surface = adjacent_surface.get()
        adj_face_id = clean_string(adjacent_surface.nameString())
        adj_room_id = None
        if adjacent_surface.space().is_initialized():
            adj_room_id = clean_string(adjacent_surface.space().get().nameString())
            bc = Surface((adj_face_id, adj_room_id), sub_face=False)
    elif os_surface.isGroundSurface():
        bc = boundary_conditions.ground
    elif surface_bc == 'Adiabatic':
        bc = Adiabatic()
    elif surface_bc == 'OtherSideCoefficients':
        temperature, htc = autocalculate, 0
        if os_surface.surfacePropertyOtherSideCoefficients().is_initialized():
            srf_prop = os_surface.surfacePropertyOtherSideCoefficients().get()
            if not srf_prop.isConstantTemperatureDefaulted():
                temperature = srf_prop.constantTemperature()
            if srf_prop.combinedConvectiveRadiativeFilmCoefficient().is_initialized():
                htc = srf_prop.combinedConvectiveRadiativeFilmCoefficient().get()
        bc = OtherSideTemperature(temperature, htc)
    if bc is None:  # set it to outdoors
        sun_exposure = True if os_surface.sunExposure() == 'SunExposed' else False
        wind_exposure = True if os_surface.windExposure() == 'WindExposed' else False
        view_factor = os_surface.viewFactortoGround()
        view_factor = view_factor.get() \
            if view_factor.is_initialized() else autocalculate
        bc = Outdoors(sun_exposure, wind_exposure, view_factor)

    # assign the construction if it exists
    if constructions is not None and not os_surface.isConstructionDefaulted():
        construction = os_surface.construction()
        if construction.is_initialized():
            const_name = clean_ep_string(construction.get().nameString())
            if const_name in constructions:
                con = constructions[const_name]
                if isinstance(con, GLASS_CONSTR):
                    face.properties.energy.construction = con

    # loop through the sub faces and convert them to Apertures and Doors
    for os_sub_surface in os_surface.subSurfaces():
        sub_surface_type = os_sub_surface.subSurfaceType()
        if 'Door' in sub_surface_type:
            door = door_from_openstudio(
                os_sub_surface, os_site_transform, constructions)
            face.add_door(door)
        else:
            ap = aperture_from_openstudio(
                os_sub_surface, os_site_transform, constructions)
            face.add_aperture(ap)
    return face


def room_from_openstudio(os_space, constructions=None, schedules=None):
    """Convert an OpenStudio Space into a Honeybee Room.

    Args:
        os_space: An OpenStudio Space to be converted into a Honeybee Room.
        constructions: An optional dictionary of Honeybee Construction objects
            which will be used to assign a construction to the room.
        schedules: An optional dictionary of Honeybee Schedule objects which will
            be used to assign schedules to the rooms.

    Returns:
        A honeybee Room object.
    """
    # translate the geometry and the room object
    os_site_transform = os_space.siteTransformation()
    faces = []
    os_surfaces = os_space.surfaces if sys.version_info < (3, 0) else os_space.surfaces()
    for os_surface in os_surfaces:
        face = face_from_openstudio(os_surface, os_site_transform, constructions)
        faces.append(face)
    room_id = clean_string(os_space.nameString())
    if room_id.endswith('_Space'):
        room_id = room_id[:-6]
    room = Room(room_id, faces, tolerance=0.01, angle_tolerance=1.0)

    # assign the display name, multiplier, story, and zone
    if os_space.displayName().is_initialized():
        room.display_name = os_space.displayName().get()
    if os_space.multiplier() != 1:
        room.multiplier = os_space.multiplier()
    inc_flr = os_space.partofTotalFloorArea if sys.version_info < (3, 0) \
        else os_space.partofTotalFloorArea()
    if not inc_flr:
        room.exclude_floor_area = True
    if os_space.buildingStory().is_initialized():
        room.story = os_space.buildingStory().get().nameString()
    if os_space.thermalZone().is_initialized():
        room.zone = os_space.thermalZone().get().nameString()

    # load any shades and assign them to the room
    shades = []
    for os_shade_group in os_space.shadingSurfaceGroups():
        shades.extend(shades_from_openstudio(os_shade_group, constructions, schedules))
    room.add_outdoor_shades(shades)

    # apply all of the loads
    if schedules is not None:
        # assign people
        for os_people in os_space.people():
            people_def = os_people.peopleDefinition()
            if people_def.peopleperSpaceFloorArea().is_initialized():
                room.properties.energy.people = \
                    people_from_openstudio(os_people, schedules)
        # assign lighting
        for os_lights in os_space.lights():
            light_def = os_lights.lightsDefinition()
            if light_def.wattsperSpaceFloorArea().is_initialized():
                room.properties.energy.lighting = \
                    lighting_from_openstudio(os_lights, schedules)
        # assign electric equipment
        for os_equip in os_space.electricEquipment():
            electric_eq_def = os_equip.electricEquipmentDefinition()
            if electric_eq_def.wattsperSpaceFloorArea().is_initialized():
                room.properties.energy.electric_equipment = \
                    electric_equipment_from_openstudio(os_equip, schedules)
        # assign gas equipment
        for os_equip in os_space.gasEquipment():
            electric_eq_def = os_equip.gasEquipmentDefinition()
            if electric_eq_def.wattsperSpaceFloorArea().is_initialized():
                room.properties.energy.gas_equipment = \
                    gas_equipment_from_openstudio(os_equip, schedules)
        # assign hot water
        if sys.version_info >= (3, 0):
            room.properties.energy.service_hot_water = hot_water_from_openstudio(
                os_space.waterUseEquipment(), room.floor_area, schedules)
        # assign process loads
        process_loads = []
        for os_other_eq in os_space.otherEquipment():
            other_eq_def = os_other_eq.otherEquipmentDefinition()
            if other_eq_def.designLevel.empty().is_initialized():
                p_load = process_from_openstudio(other_eq_def, schedules)
                process_loads.append(p_load)
        if len(process_loads) != 0:
            room.properties.energy.process_loads = process_loads
        # assign infiltration
        for os_inf in os_space.spaceInfiltrationDesignFlowRates():
            if os_inf.flowperExteriorSurfaceArea().is_initialized():
                room.properties.energy.infiltration = \
                    infiltration_from_openstudio(os_inf, schedules)
        # assign ventilation
        if os_space.designSpecificationOutdoorAir().is_initialized():
            os_vent = os_space.designSpecificationOutdoorAir().get()
            room.properties.energy.ventilation = \
                ventilation_from_openstudio(os_vent, schedules)
        # assign setpoint
        if os_space.thermalZone().is_initialized():
            os_zone = os_space.thermalZone().get()
            if os_zone.thermostatSetpointDualSetpoint().is_initialized():
                os_thermostat = os_zone.thermostatSetpointDualSetpoint().get()
                setpoint = setpoint_from_openstudio_thermostat(os_thermostat, schedules)
                if os_zone.zoneControlHumidistat().is_initialized():
                    os_humidistat = os_zone.zoneControlHumidistat().get()
                    setpoint = setpoint_from_openstudio_humidistat(
                        os_humidistat, setpoint, schedules)
                room.properties.energy.setpoint = setpoint
        # assign daylight
        if os_vector_len(os_space.daylightingControls()) != 0:
            os_daylight = os_space.daylightingControls()[0]
            room.properties.energy.daylighting_control = \
                daylight_from_openstudio(os_daylight)
    return room


def model_from_openstudio(os_model, reset_properties=False):
    """Convert an OpenStudio Model into a Honeybee Model.

    Args:
        os_model: An OpenStudio Model to be converted into a Honeybee Model.
        reset_properties: Boolean to note whether all energy properties should
            be reset to defaults upon import, meaning that only the geometry and
            boundary conditions are imported from the Openstudio Model. This
            can be particularly useful when importing an openStudio Model that
            originated from an IDF or gbXML since these formats don't support
            higher-level objects like SpaceTypes or ConstructionSets. So it is
            often easier to just import the geometry and reassign properties
            rather than working from a model where all properties are assigned
            to individual objects. (Default: False)

    Returns:
        A honeybee Model.
    """
    # load all of the energy properties from the model
    if reset_properties:
        schedules, constructions = None, None
        construction_sets, program_types = None, None
    else:
        schedules = extract_all_schedules(os_model)
        constructions = extract_all_constructions(os_model, schedules)
        construction_sets = {}
        for os_cons_set in os_model.getDefaultConstructionSets():
            if os_cons_set.nameString() != 'Default Generic Construction Set':
                con_set = construction_set_from_openstudio(os_cons_set, constructions)
                construction_sets[con_set.identifier] = con_set
        program_types = {}
        for os_space_type in os_model.getSpaceTypes():
            program = program_type_from_openstudio(os_space_type, schedules)
            program_types[program.identifier] = program

    # load all of the rooms
    rooms, zone_map = [], {}
    for os_space in os_model.getSpaces():
        room = room_from_openstudio(os_space, constructions, schedules)
        if construction_sets is not None and \
                os_space.defaultConstructionSet().is_initialized():
            os_con_set = os_space.defaultConstructionSet().get()
            try:
                room.properties.energy.construction_set = \
                    construction_sets[os_con_set.nameString()]
            except KeyError:
                pass
        if program_types is not None:
            os_space_type = os_space.spaceType if sys.version_info < (3, 0) \
                else os_space.spaceType()
            if os_space_type.is_initialized():
                os_space_type = os_space_type.get()
                try:
                    room.properties.energy.program_type = \
                        program_types[os_space_type.nameString()]
                except KeyError:
                    pass
        if os_space.thermalZone().is_initialized():
            zone_id = os_space.thermalZone().get().nameString()
            try:
                zone_map[zone_id].append(room)
            except KeyError:  # first Room in the zone
                zone_map[zone_id] = [room]
        rooms.append(room)

    # assign ideal air systems to any relevant zones
    if schedules is not None:
        for os_hvac in os_model.getZoneHVACIdealLoadsAirSystems():
            if os_hvac.thermalZone().is_initialized():
                zone_id = os_hvac.thermalZone().get().nameString()
                hvac = ideal_air_system_from_openstudio(os_hvac, schedules)
                for room in zone_map[zone_id]:
                    room.properties.energy.hvac = hvac

    # load all of the shades
    shades = []
    for os_shade_group in os_model.getShadingSurfaceGroups():
        shading_surface_type = os_shade_group.shadingSurfaceType()
        if shading_surface_type == 'Site' or shading_surface_type == 'Building':
            grp_shades = shades_from_openstudio(os_shade_group, constructions, schedules)
            shades.extend(grp_shades)

    # create the model and return it
    os_building = os_model.getBuilding()
    model_name = os_building.nameString()
    model = Model(clean_string(model_name), rooms=rooms, orphaned_shades=shades,
                  units='Meters', tolerance=0.01, angle_tolerance=1.0)
    model.display_name = model_name
    return model


def model_from_osm(osm_str, reset_properties=False, print_warnings=False):
    """Translate an OSM string to a Honeybee Model.

    Args:
        osm_str: Text string for the contents of an OSM to be converted to a
            Honeybee Model.
        reset_properties: Boolean to note whether all energy properties should be
            reset to defaults upon import, meaning that only the geometry and boundary
            conditions are imported from the Openstudio Model. (Default: False).
        print_warnings: Boolean for whether warnings about unimported objects
            should be printed. (Default: False).
    """
    # get the version translator
    if (sys.version_info < (3, 0)):
        ver_translator = openstudio.VersionTranslator()
    else:
        ver_translator = openstudio.osversion.VersionTranslator()
    os_model = ver_translator.loadModelFromString(osm_str)
    # print errors and warnings from the translation process
    if not os_model.is_initialized():
        errors = '\n'.join(str(err.logMessage()) for err in ver_translator.errors())
        raise ValueError('Failed to load model from OSM.\n{}'.format(errors))
    if print_warnings:
        for warn in ver_translator.warnings():
            print(warn.logMessage())
    # translate the OpenStudio Model to Honeybee
    return model_from_openstudio(os_model.get(), reset_properties)


def model_from_osm_file(osm_file, reset_properties=False, print_warnings=False):
    """Translate an OSM file to a Honeybee Model.

    Args:
        osm_file: Text string for the path to the OSM file to be converted to a
            Honeybee Model.
        reset_properties: Boolean to note whether all energy properties should be
            reset to defaults upon import, meaning that only the geometry and boundary
            conditions are imported from the Openstudio Model. (Default: False).
        print_warnings: Boolean for whether warnings about unimported objects
            should be printed. (Default: False).
    """
    # get the version translator
    assert os.path.isfile(osm_file), 'No file was found at: {}.'.format(osm_file)
    if (sys.version_info < (3, 0)):
        ver_translator = openstudio.VersionTranslator()
    else:
        ver_translator = openstudio.osversion.VersionTranslator()
    os_model = ver_translator.loadModel(os_path(osm_file))
    # print errors and warnings from the translation process
    if not os_model.is_initialized():
        errors = '\n'.join(str(err.logMessage()) for err in ver_translator.errors())
        raise ValueError('Failed to load model from OSM.\n{}'.format(errors))
    if print_warnings:
        for warn in ver_translator.warnings():
            print(warn.logMessage())
    # translate the OpenStudio Model to Honeybee
    return model_from_openstudio(os_model.get(), reset_properties)


def model_from_idf_file(idf_file, reset_properties=False, print_warnings=False):
    """Translate an IDF file to a Honeybee Model.

    Args:
        idf_file: Text string for the path to the IDF file to be converted to a
            Honeybee Model.
        reset_properties: Boolean to note whether all energy properties should be
            reset to defaults upon import, meaning that only the geometry and boundary
            conditions are imported from the Openstudio Model. (Default: False).
        print_warnings: Boolean for whether warnings about unimported objects
            should be printed. (Default: False).
    """
    # get the version translator
    assert os.path.isfile(idf_file), 'No file was found at: {}.'.format(idf_file)
    if (sys.version_info < (3, 0)):
        idf_translator = openstudio.EnergyPlusReverseTranslator()
    else:
        idf_translator = openstudio.energyplus.ReverseTranslator()
    os_model = idf_translator.loadModel(os_path(idf_file))
    # print errors and warnings from the translation process
    if not os_model.is_initialized():
        errors = '\n'.join(str(err.logMessage()) for err in idf_translator.errors())
        raise ValueError('Failed to load model from IDF.\n{}'.format(errors))
    if print_warnings:
        for warn in idf_translator.warnings():
            print(warn.logMessage())
    # translate the OpenStudio Model to Honeybee
    return model_from_openstudio(os_model.get(), reset_properties)


def model_from_gbxml_file(gbxml_file, reset_properties=False, print_warnings=False):
    """Translate a gbXML file to a Honeybee Model.

    Args:
        gbxml_file: Text string for the path to the gbXML file to be converted
            to a Honeybee Model.
        reset_properties: Boolean to note whether all energy properties should be
            reset to defaults upon import, meaning that only the geometry and boundary
            conditions are imported from the Openstudio Model. (Default: False).
        print_warnings: Boolean for whether warnings about unimported objects
            should be printed. (Default: False).
    """
    # get the version translator
    assert os.path.isfile(gbxml_file), 'No file was found at: {}.'.format(gbxml_file)
    if (sys.version_info < (3, 0)):
        gbxml_translator = openstudio.GbXMLReverseTranslator()
    else:
        gbxml_translator = openstudio.gbxml.GbXMLReverseTranslator()
    os_model = gbxml_translator.loadModel(os_path(gbxml_file))
    # print errors and warnings from the translation process
    if not os_model.is_initialized():
        errors = '\n'.join(str(err.logMessage()) for err in gbxml_translator.errors())
        raise ValueError('Failed to load model from OSM.\n{}'.format(errors))
    if print_warnings:
        for warn in gbxml_translator.warnings():
            print(warn.logMessage())
    # remove any shade groups that were translated as spaces
    os_model = os_model.get()
    os_spaces = os_model.getSpaces()
    for os_space in os_spaces:
        os_surfaces = os_space.surfaces if sys.version_info < (3, 0) \
            else os_space.surfaces()
        if os_vector_len(os_surfaces) == 0:
            os_space.remove()
    # translate the OpenStudio Model to Honeybee
    return model_from_openstudio(os_model, reset_properties)
