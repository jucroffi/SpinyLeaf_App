# coding=utf-8
"""Methods to write to inp."""
from __future__ import division
import os
import math

from ladybug_geometry.geometry2d import Vector2D, Point2D
from ladybug_geometry.geometry3d import Vector3D, Point3D, LineSegment3D, Plane, Face3D
from ladybug_geometry.bounding import bounding_box
from honeybee.typing import clean_doe2_string, clean_string
from honeybee.boundarycondition import Surface
from honeybee.facetype import Wall, Floor, RoofCeiling
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.air import AirBoundaryConstruction
from honeybee_energy.lib.constructionsets import generic_construction_set

from .config import DOE2_TOLERANCE, DOE2_ANGLE_TOL, GEO_DEC_COUNT, RECT_WIN_SUBD, \
    DOE2_INTERIOR_BCS, GEO_CHARS, RES_CHARS
from .util import generate_inp_string, header_comment_minor, \
    header_comment_major, switch_statement_id
from .grouping import group_rooms_by_doe2_level, group_rooms_by_doe2_hvac
from .construction import opaque_material_to_inp, opaque_construction_to_inp, \
    window_construction_to_inp, door_construction_to_inp, air_construction_to_inp
from .schedule import energy_trans_sch_to_transmittance
from .load import people_to_inp, lighting_to_inp, electric_equipment_to_inp, \
    hot_water_and_gas_to_inp, infiltration_to_inp, setpoint_to_inp, ventilation_to_inp
from .programtype import program_type_to_inp, switch_dict_to_space_inp, \
    switch_dict_to_zone_inp
from .simulation import SimulationPar


def face_3d_to_inp(face_3d, parent_name='HB object'):
    """Convert a Face3D into a DOE-2 POLYGON string and info to position it in space.

    In this operation, all holes in the Face3D are ignored since they are not
    supported by DOE-2. Collapsing the boundary and holes into a single list
    that winds inward to cut out the holes will cause eQuest to raise an error.

    Args:
        face_3d: A ladybug-geometry Face3D object for which a INP POLYGON
            string will be generated.
        parent_name: The name of the parent object that will reference this
            POLYGON. This will be used to generate a name for the polygon.
            Note that this should ideally have 24 characters or less so that
            the result complies with the strict 32 character limit of DOE-2
            identifiers.

    Returns:
        A tuple with two elements.

        -   polygon_str: Text string for the INP polygon.

        -   position_info: A tuple of values used to locate the Polygon in 3D space.
            The order of properties in the tuple is as follows: (ORIGIN, TILT, AZIMUTH).
    """
    # TODO: Consider adding a workaround for the DOE-2 limit of 40 vertices
    # perhaps we can just say NO-SHAPE and specify AREA, VOLUME, and HEIGHT
    # get the main properties that place the geometry in 3D space
    pts_3d = face_3d.lower_left_counter_clockwise_boundary
    tilt, azimuth = math.degrees(face_3d.tilt), math.degrees(face_3d.azimuth)
    llc_origin = face_3d.lower_left_corner
    llc_coords = []
    for coord in llc_origin:  # avoid signed zero
        coord = round(coord, GEO_DEC_COUNT)
        clean_coord = 0.0 if coord == 0 else coord
        llc_coords.append(clean_coord)
    llc_origin = Point3D.from_array(llc_coords)

    # get the 2D vertices in the plane of the Face
    if DOE2_ANGLE_TOL <= tilt <= 180 - DOE2_ANGLE_TOL:  # vertical or tilted
        proj_y = Vector3D(0, 0, 1).project(face_3d.normal)
        proj_x = proj_y.rotate(face_3d.normal, math.pi / -2)
        ref_plane = Plane(face_3d.normal, llc_origin, proj_x)
        vertices = [ref_plane.xyz_to_xy(pt) for pt in pts_3d]
    else:  # horizontal; ensure vertices are always counterclockwise from above
        azimuth = 180.0
        llc = Point2D(llc_origin.x, llc_origin.y)
        vertices = [Point2D(v.x - llc.x, v.y - llc.y) for v in pts_3d]
        if tilt > 180 - DOE2_ANGLE_TOL:
            vertices = [Point2D(v.x, -v.y) for v in vertices]

    # format the vertices into a POLYGON string
    verts_values = []
    for pt in vertices:
        x_coord = round(pt.x, GEO_DEC_COUNT)
        y_coord = round(pt.y, GEO_DEC_COUNT)
        if x_coord == 0:  # avoid signed zero
            x_coord = 0.0
        if y_coord == 0:  # avoid signed zero
            y_coord = 0.0
        verts_values.append('({}, {})'.format(x_coord, y_coord))
    verts_keywords = tuple('V{}'.format(i + 1) for i in range(len(verts_values)))
    poly_name = '{} Plg'.format(parent_name)
    polygon_str = generate_inp_string(poly_name, 'POLYGON', verts_keywords, verts_values)
    position_info = (llc_origin, tilt, azimuth)
    return polygon_str, position_info


def face_3d_to_inp_rectangle(face_3d):
    """Convert a Face3D into parameters needed to represent it as a rectangle in INP.

    The output of this function will be None if the Face3D cannot be represented
    as an INP rectangle without alteration of the geometry.

    Args:
        face_3d: A ladybug-geometry Face3D object which will be tested for whether
            it can be represented as a rectangle in INP.

    Returns:
        Will be None if the Face3D cannot be translated to a WIDTH and HEIGHT
        without alteration of the geometry. If the geometry can be successfully
        translated, this will be a tuple with five elements.

        -   width: A number for the width of the rectangle.

        -   height: A number for the height of the rectangle.

        -   llc_origin: A Point3D for the lower-left corner of the Shade
            geometry origin.

        -   tilt: A number for the tilt of the rectangle in degrees.

        -   azimuth: A number for the azimuth of the rectangle in degrees.
    """
    if face_3d.boundary_polygon2d.is_rectangle(math.radians(DOE2_ANGLE_TOL)):
        # check to see at least one of the segments is horizontal
        are_segs_hor = [seg.max.z - seg.min.z <= DOE2_TOLERANCE
                        for seg in face_3d.boundary_segments]
        if True in are_segs_hor:
            pts_3d = face_3d.lower_left_counter_clockwise_boundary
            llc_origin = pts_3d[0]
            width = llc_origin.distance_to_point(pts_3d[1])
            height = llc_origin.distance_to_point(pts_3d[-1])
            if all(is_horiz for is_horiz in are_segs_hor):  # horizontal; adjust azimuth
                tilt = 0.0
                hgt_vec = llc_origin - pts_3d[-1]
                hgt_vec_2d = Vector2D(hgt_vec.x, hgt_vec.y)
                azimuth = math.degrees(Vector2D(0, 1).angle_clockwise(hgt_vec_2d))
            else:  # vertical or tilted; use Face3D azimuth
                tilt = math.degrees(face_3d.tilt)
                azimuth = math.degrees(face_3d.azimuth)
            return width, height, llc_origin, tilt, azimuth
    return None


def shade_mesh_to_inp(shade_mesh, equest_version=None):
    """Generate an INP string representation of a ShadeMesh.

    Args:
        shade_mesh: A honeybee ShadeMesh for which an INP representation
            will be returned.
        equest_version: An optional text string to denote the version of eQuest
            for which the Shade INP definition will be generated. If unspecified
            or unrecognized, the latest version of eQuest will be used.

    Returns:
        A tuple with two elements.

        -   shade_polygons: A list of text strings for the INP polygons needed
            to represent the ShadeMesh.

        -   shade_defs: A list of text strings for the INP definitions needed
            to represent the ShadeMesh.
    """
    # extract the transmittance properties of the shade
    base_id = clean_doe2_string(shade_mesh.identifier, GEO_CHARS)
    trans_kwd = ['TRANSMITTANCE']
    trans_vals = [energy_trans_sch_to_transmittance(shade_mesh)]
    t_sch_obj = shade_mesh.properties.energy.transmittance_schedule
    if t_sch_obj is not None and not t_sch_obj.is_constant:
        trans_kwd.append('SHADE-SCHEDULE')
        t_shc_id = clean_doe2_string(t_sch_obj.identifier, RES_CHARS)
        trans_vals.append('"{}"'.format(t_shc_id))

    # set up collector lists and properties for all shades
    shade_polygons, shade_defs = [], []

    # loop through the mesh faces and create individual shade objects
    for i, face in enumerate(shade_mesh.geometry.face_vertices):
        doe2_id = '{}{}'.format(base_id, i)
        f_geo = Face3D(face)
        shd_geo = f_geo if f_geo.altitude > 0 else f_geo.flip()
        clean_geo = shd_geo.remove_colinear_vertices(DOE2_TOLERANCE)
        rect_info = face_3d_to_inp_rectangle(clean_geo)
        if equest_version == '3.64':
            shade_polygon = ''
            if rect_info is not None:
                width, height, origin, tilt, az = rect_info
            else:  # take the bounding rectangle around the Face3D
                min_pt, max_pt = clean_geo.min, clean_geo.max
                f_tilt = math.degrees(clean_geo.tilt)
                if 90 - DOE2_ANGLE_TOL <= f_tilt <= 90 + DOE2_ANGLE_TOL:  # vertical
                    seg_dir = Vector3D(max_pt.x - min_pt.x, max_pt.y - min_pt.y, 0)
                    seg = LineSegment3D(min_pt, seg_dir)
                    ext_dir = Vector3D(0, 0, max_pt.z - min_pt.z)
                else:  # horizontal or tilted
                    seg = LineSegment3D(min_pt, Vector3D(max_pt.x - min_pt.x, 0, 0))
                    ext_dir = Vector3D(0, max_pt.y - min_pt.y, max_pt.z - min_pt.z)
                rect_geo = Face3D.from_extrusion(seg, ext_dir)
                width, height, origin, tilt, az = face_3d_to_inp_rectangle(rect_geo)
            geo_kwd, geo_vals = ['HEIGHT', 'WIDTH'], [height, width]
        elif rect_info is not None:  # shade is a rectangle; translate it without POLYGON
            width, height, origin, tilt, az = rect_info
            geo_kwd = ['SHAPE', 'HEIGHT', 'WIDTH']
            geo_vals = ['RECTANGLE', height, width]
        else:  # otherwise, create the polygon string from the geometry
            shade_polygon, pos_info = face_3d_to_inp(clean_geo, doe2_id)
            shade_polygons.append(shade_polygon)
            origin, tilt, az = pos_info
            geo_kwd = ['SHAPE', 'POLYGON']
            geo_vals = ['POLYGON', '"{} Plg"'.format(doe2_id)]
        geo_kwd.extend(('X-REF', 'Y-REF', 'Z-REF', 'TILT', 'AZIMUTH'))
        geo_vals.extend((round(origin.x, GEO_DEC_COUNT), round(origin.y, GEO_DEC_COUNT),
                        round(origin.z, GEO_DEC_COUNT), tilt, az))
        # create the final shade definition, which includes the position information
        keywords = geo_kwd + trans_kwd
        values = geo_vals + trans_vals
        shade_def = generate_inp_string(doe2_id, 'FIXED-SHADE', keywords, values)
        shade_defs.append(shade_def)

    return shade_polygons, shade_defs


def shade_to_inp(shade, equest_version=None):
    """Generate an INP string representation of a Shade.

    Args:
        shade: A honeybee Shade for which an INP representation will be returned.
        equest_version: An optional text string to denote the version of eQuest
            for which the Shade INP definition will be generated. If unspecified
            or unrecognized, the latest version of eQuest will be used.

    Returns:
        A tuple with two elements.

        -   shade_polygon: Text string for the INP polygon for the Shade.

        -   shade_def: Text string for the INP definition of the Shade.
    """
    # extract the transmittance properties of the shade
    doe2_id = clean_doe2_string(shade.identifier, GEO_CHARS)
    trans_kwd = ['TRANSMITTANCE']
    trans_vals = [energy_trans_sch_to_transmittance(shade)]
    t_sch_obj = shade.properties.energy.transmittance_schedule
    if t_sch_obj is not None and not t_sch_obj.is_constant:
        trans_kwd.append('SHADE-SCHEDULE')
        t_shc_id = clean_doe2_string(t_sch_obj.identifier, RES_CHARS)
        trans_vals.append('"{}"'.format(t_shc_id))

    # extract the geometry properties of the shade
    shd_geo = shade.geometry if shade.altitude > 0 else shade.geometry.flip()
    clean_geo = shd_geo.remove_colinear_vertices(DOE2_TOLERANCE)
    rect_info = face_3d_to_inp_rectangle(clean_geo)
    if equest_version == '3.64':
        shade_polygon = ''
        if rect_info is not None:
            width, height, origin, tilt, az = rect_info
        else:  # take the bounding rectangle around the Face3D
            min_pt, max_pt = clean_geo.min, clean_geo.max
            f_tilt = math.degrees(clean_geo.tilt)
            if 90 - DOE2_ANGLE_TOL <= f_tilt <= 90 + DOE2_ANGLE_TOL:  # vertical
                seg_dir = Vector3D(max_pt.x - min_pt.x, max_pt.y - min_pt.y, 0)
                seg = LineSegment3D(min_pt, seg_dir)
                ext_dir = Vector3D(0, 0, max_pt.z - min_pt.z)
            else:  # horizontal or tilted
                seg = LineSegment3D(min_pt, Vector3D(max_pt.x - min_pt.x, 0, 0))
                ext_dir = Vector3D(0, max_pt.y - min_pt.y, max_pt.z - min_pt.z)
            rect_geo = Face3D.from_extrusion(seg, ext_dir)
            width, height, origin, tilt, az = face_3d_to_inp_rectangle(rect_geo)
        geo_kwd, geo_vals = ['HEIGHT', 'WIDTH'], [height, width]
    elif rect_info is not None:  # shade is a rectangle; translate it without POLYGON
        width, height, origin, tilt, az = rect_info
        geo_kwd = ['SHAPE', 'HEIGHT', 'WIDTH']
        geo_vals = ['RECTANGLE', height, width]
        shade_polygon = ''
    else:  # otherwise, create the polygon string from the geometry
        shade_polygon, pos_info = face_3d_to_inp(clean_geo, doe2_id)
        origin, tilt, az = pos_info
        geo_kwd = ['SHAPE', 'POLYGON']
        geo_vals = ['POLYGON', '"{} Plg"'.format(doe2_id)]
    geo_kwd.extend(('X-REF', 'Y-REF', 'Z-REF', 'TILT', 'AZIMUTH'))
    geo_vals.extend((round(origin.x, GEO_DEC_COUNT), round(origin.y, GEO_DEC_COUNT),
                     round(origin.z, GEO_DEC_COUNT), tilt, az))

    # create the final shade definition, which includes the position information
    keywords = geo_kwd + trans_kwd
    values = geo_vals + trans_vals
    shade_def = generate_inp_string(doe2_id, 'FIXED-SHADE', keywords, values)
    return shade_polygon, shade_def


def door_to_inp(door):
    """Generate an INP string representation of a Door.

    Doors assigned to a parent Face will use the parent Face plane in order to
    determine their XY coordinates. Otherwise, the Door's own plane will be used.

    Note that the resulting string does not include full construction definitions.
    Also note that shades assigned to the Door are not included in the resulting
    string. To write these objects into a final string, you must loop through the
    Door.shades, and call the to.inp method on each one.

    Args:
        door: A honeybee Door for which an INP representation will be returned.

    Returns:
        Text string for the INP definition of the Door.
    """
    # extract the plane information from the parent geometry
    if door.has_parent:
        parent_llc = door.parent.geometry.lower_left_corner
        rel_plane = door.parent.geometry.plane
    else:
        parent_llc = door.geometry.lower_left_corner
        rel_plane = door.geometry.plane
    # get the LLC and URC of the bounding rectangle of the door
    apt_llc = door.geometry.lower_left_corner
    apt_urc = door.geometry.upper_right_corner

    # determine the width and height and origin in the parent coordinate system
    if DOE2_ANGLE_TOL <= door.tilt <= 180 - DOE2_ANGLE_TOL:  # vertical or tilted
        proj_y = Vector3D(0, 0, 1).project(rel_plane.n)
        proj_x = proj_y.rotate(rel_plane.n, math.pi / -2)
    else:  # located within the XY plane
        proj_x = Vector3D(1, 0, 0)
    ref_plane = Plane(rel_plane.n, parent_llc, proj_x)
    min_2d = ref_plane.xyz_to_xy(apt_llc)
    max_2d = ref_plane.xyz_to_xy(apt_urc)
    width = round(max_2d.x - min_2d.x, GEO_DEC_COUNT)
    height = round(max_2d.y - min_2d.y, GEO_DEC_COUNT)

    # create the aperture definition
    doe2_id = clean_doe2_string(door.identifier, GEO_CHARS)
    dr_con = door.properties.energy.construction
    constr_o_name = dr_con.identifier if isinstance(dr_con, OpaqueConstruction) \
        else dr_con.identifier + '_d'
    constr = clean_doe2_string(constr_o_name, RES_CHARS)
    keywords = ('X', 'Y', 'WIDTH', 'HEIGHT', 'CONSTRUCTION')
    values = (round(min_2d.x, GEO_DEC_COUNT), round(min_2d.y, GEO_DEC_COUNT),
              width, height, '"{}"'.format(constr))
    door_def = generate_inp_string(doe2_id, 'DOOR', keywords, values)
    return door_def


def aperture_to_inp(aperture):
    """Generate an INP string representation of a Aperture.

    Apertures assigned to a parent Face will use the parent Face plane in order to
    determine their XY coordinates. Otherwise, the Aperture's own plane will be used.

    Note that the resulting string does not include full construction definitions.
    Also note that shades assigned to the Aperture are not included in the resulting
    string. To write these objects into a final string, you must loop through the
    Aperture.shades, and call the to.inp method on each one.

    Args:
        aperture: A honeybee Aperture for which an INP representation will be returned.

    Returns:
        Text string for the INP definition of the Aperture.
    """
    # extract the plane information from the parent geometry
    if aperture.has_parent:
        parent_llc = aperture.parent.geometry.lower_left_corner
        rel_plane = aperture.parent.geometry.plane
    else:
        parent_llc = aperture.geometry.lower_left_corner
        rel_plane = aperture.geometry.plane
    # get the LLC and URC of the bounding rectangle of the aperture
    apt_llc = aperture.geometry.lower_left_corner
    apt_urc = aperture.geometry.upper_right_corner

    # determine the width and height and origin in the parent coordinate system
    if DOE2_ANGLE_TOL <= aperture.tilt <= 180 - DOE2_ANGLE_TOL:  # vertical or tilted
        proj_y = Vector3D(0, 0, 1).project(rel_plane.n)
        proj_x = proj_y.rotate(rel_plane.n, math.pi / -2)
    else:  # located within the XY plane
        proj_x = Vector3D(1, 0, 0)
    ref_plane = Plane(rel_plane.n, parent_llc, proj_x)
    min_2d = ref_plane.xyz_to_xy(apt_llc)
    max_2d = ref_plane.xyz_to_xy(apt_urc)
    width = round(max_2d.x - min_2d.x, GEO_DEC_COUNT)
    height = round(max_2d.y - min_2d.y, GEO_DEC_COUNT)

    # create the aperture definition
    doe2_id = clean_doe2_string(aperture.identifier, GEO_CHARS)
    constr_o_name = aperture.properties.energy.construction.identifier
    constr = clean_doe2_string(constr_o_name, RES_CHARS)
    keywords = ('X', 'Y', 'WIDTH', 'HEIGHT', 'GLASS-TYPE')
    values = (round(min_2d.x, GEO_DEC_COUNT), round(min_2d.y, GEO_DEC_COUNT),
              width, height, '"{}"'.format(constr))
    aperture_def = generate_inp_string(doe2_id, 'WINDOW', keywords, values)
    return aperture_def


def face_to_inp(face, space_origin=Point3D(0, 0, 0), location=None):
    """Generate an INP string representation of a Face.

    Note that the resulting string does not include full construction definitions.

    Also note that this does not include any of the shades assigned to the Face
    in the resulting string. Nor does it include the strings for the
    apertures or doors. To write these objects into a final string, you must
    loop through the Face.apertures, and Face.doors and call the to.inp method
    on each one.

    Args:
        face: A honeybee Face for which an INP representation will be returned.
        space_origin: A ladybug-geometry Point3D for the origin of the space
            to which the Face is assigned. (Default: (0, 0, 0)).
        location: An optional text string to note the DOE-2 LOCATION of the
            Face on the parent Room. When this is specified, the Face will be
            written without using a POLYGON. (Default: None).

    Returns:
        A tuple with two elements.

        -   face_polygon: Text string for the INP polygon for the Face.

        -   face_def: Text string for the INP definition of the Face.
    """
    # set up attributes based on the face type and boundary condition
    f_type_str, bc_str = str(face.type), str(face.boundary_condition)
    if bc_str == 'Outdoors':
        doe2_type = 'EXTERIOR-WALL'  # DOE2 uses walls for a lot of things
        if f_type_str == 'RoofCeiling':
            doe2_type = 'ROOF'
    elif bc_str in DOE2_INTERIOR_BCS or f_type_str == 'AirBoundary':
        doe2_type = 'INTERIOR-WALL'  # DOE2 uses walls for a lot of things
    else:  # likely ground or some other fancy ground boundary condition
        doe2_type = 'UNDERGROUND-WALL'

    # process the face identifier and the construction
    doe2_id = clean_doe2_string(face.identifier, GEO_CHARS)
    constr_o_name = face.properties.energy.construction.identifier
    constr = clean_doe2_string(constr_o_name, RES_CHARS)

    # process the geometry
    if location is not None:
        keywords = ['CONSTRUCTION', 'LOCATION']
        values = ['"{}"'.format(constr), location]
        face_polygon = ''
    else:  # create the polygon string from the geometry
        f_geo = face.geometry.remove_colinear_vertices(DOE2_TOLERANCE)
        face_polygon, pos_info = face_3d_to_inp(f_geo, doe2_id)
        face_origin, tilt, az = pos_info
        origin = face_origin - space_origin
        keywords = ['POLYGON', 'CONSTRUCTION', 'TILT', 'AZIMUTH', 'X', 'Y', 'Z']
        values = ['"{} Plg"'.format(doe2_id), '"{}"'.format(constr), tilt, az,
                  round(origin.x, GEO_DEC_COUNT),
                  round(origin.y, GEO_DEC_COUNT),
                  round(origin.z, GEO_DEC_COUNT)]

    # add information related to the boundary condition
    if bc_str == 'Surface':
        adj_room = face.boundary_condition.boundary_condition_objects[-1]
        adj_id = clean_doe2_string(adj_room, GEO_CHARS)
        values.append('"{}"'.format(adj_id))
        keywords.append('NEXT-TO')
    elif doe2_type == 'INTERIOR-WALL':  # assume that it is adiabatic
        keywords.append('INT-WALL-TYPE')
        values.append('ADIABATIC')
    if location is None and f_type_str == 'Floor' and doe2_type != 'INTERIOR-WALL':
        keywords.append('LOCATION')
        values.append('BOTTOM')

    # create the face definition
    face_def = generate_inp_string(doe2_id, doe2_type, keywords, values)
    return face_polygon, face_def


def room_to_inp(room, floor_origin=Point3D(0, 0, 0), floor_height=None,
                exclude_interior_walls=False, exclude_interior_ceilings=False):
    """Generate an INP string representation of a Room.

    This will include the Room's constituent Faces, Apertures and Doors with
    each of these elements being a separate item in the list of strings returned.
    However, any shades assigned to the Room or its constituent elements are
    excluded and should be written by looping through the shades on the parent model.

    The resulting string will also include all internal gain definitions for the
    Room (people, lights, equipment), infiltration definitions, ventilation
    requirements, and thermostat objects.

    However, complete schedule definitions assigned to these load objects are
    excluded as well as any construction or material definitions.

    Args:
        floor_origin: A ladybug-geometry Point3D for the origin of the
            floor (aka. story) to which the Room is a part of. (Default: (0, 0, 0)).
        floor_height: An optional number for the parent story SPACE-HEIGHT,
            which will be used to check the Room geometry to determine if
            it must be written using POLYGONs. If None, no check will be
            performed. (Default: None)
        exclude_interior_walls: Boolean to note whether interior wall Faces
            should be excluded from the resulting string. (Default: False).
        exclude_interior_ceilings: Boolean to note whether interior ceiling
            Faces should be excluded from the resulting string. (Default: False).

    Returns:
        A tuple with two elements.

        -   room_polygons: A list of text strings for the INP polygons needed
            to represent the Room and all of its constituent Faces.

        -   room_defs: A list of text strings for the INP definitions needed
            to represent the Room and all of its constituent Faces, Apertures
            and Doors.
    """
    # process the room identifier
    doe2_id = clean_doe2_string(room.identifier, GEO_CHARS)

    # set up attributes based on the Room's energy properties
    energy_attr_keywords = ['ZONE-TYPE']
    energy_attr_values = [room_doe2_conditioning_type(room)]
    if room.properties.energy._program_type is not None:
        energy_attr_keywords.append('C-ACTIVITY-DESC')
        prog_uid = switch_statement_id(room.properties.energy.program_type.identifier)
        energy_attr_values.append('*{}*'.format(prog_uid))
    # people
    ppl_kwd, ppl_val = people_to_inp(room.properties.energy._people)
    energy_attr_keywords.extend(ppl_kwd)
    energy_attr_values.extend(ppl_val)
    # lighting
    lgt_kwd, lgt_val = lighting_to_inp(room.properties.energy._lighting)
    energy_attr_keywords.extend(lgt_kwd)
    energy_attr_values.extend(lgt_val)
    # equipment
    eq_kwd, eq_val = electric_equipment_to_inp(
        room.properties.energy._electric_equipment)
    energy_attr_keywords.extend(eq_kwd)
    energy_attr_values.extend(eq_val)
    # hot water and gas usage
    shw_gas_kwd, shw_gas_val = hot_water_and_gas_to_inp(
        room.properties.energy.service_hot_water,
        room.properties.energy.gas_equipment, room.floor_area)
    energy_attr_keywords.extend(shw_gas_kwd)
    energy_attr_values.extend(shw_gas_val)
    # infiltration
    inf_kwd, inf_val = infiltration_to_inp(room.properties.energy._infiltration)
    energy_attr_keywords.extend(inf_kwd)
    energy_attr_values.extend(inf_val)

    def _is_room_3d_extruded(hb_room):
        """Test if a Room is a pure extrusion.

        Args:
            hb_room: The Honeybee Room to be tested.

        Returns:
            A tuple with two elements.

            -   is_extrusion: True if the geometry is an extrusion. False if not.

            -   face_orientations: A list of integers that aligns with the Room.faces
                and denotes whether each face is downward (-1), vertical (0) or
                upward (+1).
        """
        # first check if we have to use POLYGONS because of the parent SPACE-HEIGHT
        if floor_height is not None:
            room_height = room.max.z - room.min.z
            if abs(room_height - floor_height) > DOE2_TOLERANCE:
                return False, []

        # set up the parameters for evaluating vertical or horizontal
        vert_vec = Vector3D(0, 0, 1)
        min_v_ang = math.radians(DOE2_ANGLE_TOL)
        max_v_ang = math.pi - min_v_ang
        min_h_ang = (math.pi / 2) - min_v_ang
        max_h_ang = (math.pi / 2) + min_v_ang

        # loop through the Room faces and test them
        face_orientations = []
        for face in hb_room.faces:
            try:  # first make sure that the geometry is not degenerate
                clean_geo = face.geometry.remove_colinear_vertices(DOE2_TOLERANCE)
                v_ang = clean_geo.normal.angle(vert_vec)
                if v_ang <= min_v_ang:
                    face_orientations.append(1)
                    continue
                elif v_ang >= max_v_ang:
                    face_orientations.append(-1)
                    continue
                elif min_h_ang <= v_ang <= max_h_ang:
                    face_orientations.append(0)
                    continue
                return False, []
            except AssertionError:  # degenerate face to ignore
                pass
        return True, face_orientations

    # if the room is extruded, determine the locations of each face
    face_locations = []
    is_extrusion, face_orientations = _is_room_3d_extruded(room)
    if is_extrusion:  # try to translate without using POLYGON for the Room faces
        if room.properties.doe2.space_polygon_geometry is not None:
            r_geo = room.properties.doe2.space_polygon_geometry
        else:
            try:
                r_geo = room.horizontal_boundary(
                    match_walls=True, tolerance=DOE2_TOLERANCE)
            except Exception:  # we may need to write it with NO-SHAPE
                r_geo = None
        if r_geo is not None:
            r_geo = r_geo if r_geo.normal.z >= 0 else r_geo.flip()
            r_geo = r_geo.remove_duplicate_vertices(DOE2_TOLERANCE)
            rm_pts = r_geo.lower_left_counter_clockwise_boundary
            rm_height = room.max.z - room.min.z
            ceil_count = len([orient for orient in face_orientations if orient == 1])
            floor_count = len([orient for orient in face_orientations if orient == -1])
            for face, orient in zip(room.faces, face_orientations):
                if orient == 0:  # wall to associate with a room vertex
                    clean_geo = face.geometry.remove_colinear_vertices(DOE2_TOLERANCE)
                    face_height = face.max.z - face.min.z
                    if clean_geo.boundary_polygon2d.is_rectangle(DOE2_ANGLE_TOL) and \
                            abs(rm_height - face_height) <= DOE2_TOLERANCE:
                        f_origin = face.geometry.lower_left_corner
                        for i, r_pt in enumerate(rm_pts):
                            if f_origin.is_equivalent(r_pt, DOE2_TOLERANCE):
                                face_locations.append('SPACE-V{}'.format(i + 1))
                                break
                        else:  # not associated with any Room vertex
                            face_locations.append(None)
                    else:  # not a rectangular geometry
                        face_locations.append(None)
                elif orient == 1:
                    loc = 'TOP' if ceil_count == 1 and not r_geo.has_holes else None
                    face_locations.append(loc)
                else:
                    loc = 'BOTTOM' if floor_count == 1 else None
                    face_locations.append(loc)

    # if the room is not extruded, just use the generic horizontal boundary
    if len(face_locations) == 0:
        if room.properties.doe2.space_polygon_geometry is not None:
            r_geo = room.properties.doe2.space_polygon_geometry
        else:
            try:
                r_geo = room.horizontal_boundary(
                    match_walls=False, tolerance=DOE2_TOLERANCE)
                r_geo = r_geo if r_geo.normal.z >= 0 else r_geo.flip()
                r_geo = r_geo.remove_colinear_vertices(tolerance=DOE2_TOLERANCE)
            except Exception:  # we may need to write it with NO-SHAPE
                r_geo = None
        face_locations = [None] * len(room.faces)

    # create the space definition
    if r_geo is None:   # we have to use NO-SHAPE
        msg = 'Using NO-SHAPE for SPACE "{}".'.format(room.display_name)
        print(msg)
        space_origin = room.min
        origin = space_origin - floor_origin
        keywords = ['SHAPE', 'AZIMUTH', 'X', 'Y', 'Z', 'AREA', 'VOLUME']
        values = ['NO-SHAPE', 0, round(origin.x, GEO_DEC_COUNT),
                  round(origin.y, GEO_DEC_COUNT), round(origin.z, GEO_DEC_COUNT),
                  round(room.floor_area, GEO_DEC_COUNT),
                  round(room.volume, GEO_DEC_COUNT)]
        if room.multiplier != 1:
            keywords.append('MULTIPLIER')
            values.append(room.multiplier)
        keywords.extend(energy_attr_keywords)
        values.extend(energy_attr_values)
        space_def = generate_inp_string(doe2_id, 'SPACE', keywords, values)
        room_polygons = []
        room_defs = [space_def]
    else:
        # create the room polygon string from the geometry
        room_polygon, pos_info = face_3d_to_inp(r_geo, doe2_id)
        space_origin, _, _ = pos_info
        origin = space_origin - floor_origin
        # create the space definition, which includes the position info
        keywords = ['SHAPE', 'POLYGON', 'AZIMUTH', 'X', 'Y', 'Z', 'VOLUME']
        values = ['POLYGON', '"{} Plg"'.format(doe2_id), 0,
                  round(origin.x, GEO_DEC_COUNT), round(origin.y, GEO_DEC_COUNT),
                  round(origin.z, GEO_DEC_COUNT), round(room.volume, GEO_DEC_COUNT)]
        if room.multiplier != 1:
            keywords.append('MULTIPLIER')
            values.append(room.multiplier)
        keywords.extend(energy_attr_keywords)
        values.extend(energy_attr_values)
        space_def = generate_inp_string(doe2_id, 'SPACE', keywords, values)
        room_polygons = [room_polygon]
        room_defs = [space_def]

    # gather together all face definitions and polygons to define the room
    for face, f_loc in zip(room.faces, face_locations):
        # first check if this is a face that should be excluded
        if isinstance(face.boundary_condition, Surface):
            if exclude_interior_walls and isinstance(face.type, Wall):
                continue
            elif exclude_interior_ceilings and \
                    isinstance(face.type, (Floor, RoofCeiling)):
                continue
        # add the face definition along with all apertures and doors
        face_polygon, face_def = face_to_inp(face, space_origin, f_loc)
        if face_polygon != '':
            room_polygons.append(face_polygon)
        room_defs.append(face_def)
        for ap in face.apertures:
            ap_def = aperture_to_inp(ap)
            room_defs.append(ap_def)
        if not isinstance(face.boundary_condition, Surface):
            for dr in face.doors:
                dr_def = door_to_inp(dr)
                room_defs.append(dr_def)
    return room_polygons, room_defs


def model_to_inp(
    model, simulation_par=None, hvac_mapping='Story',
    exclude_interior_walls=False, exclude_interior_ceilings=False, equest_version=None
):
    """Generate an INP string representation of a Model.

    The resulting string will include all geometry (Rooms, Faces, Apertures,
    Doors, Shades), all fully-detailed constructions + materials, all fully-detailed
    schedules, and the room properties. It will also include the simulation
    parameters. Essentially, the string includes everything needed to simulate
    the model.

    Args:
        model: A honeybee Model for which an INP representation will be returned.
        simulation_par: A honeybee-doe2 SimulationPar object to specify how the
            DOE-2 simulation should be run. If None, default simulation
            parameters will be generated, which will run the simulation for the
            full year. (Default: None).
        hvac_mapping: Text to indicate how HVAC systems should be assigned to the
            exported model. Story will assign one HVAC system for each distinct
            level polygon, Model will use only one HVAC system for the whole model
            and AssignedHVAC will follow how the HVAC systems have been assigned
            to the Rooms.properties.energy.hvac. Choose from the options
            below. (Default: Story).

            * Room
            * Story
            * Model
            * AssignedHVAC

        exclude_interior_walls: Boolean to note whether interior wall Faces
            should be excluded from the resulting string. (Default: False).
        exclude_interior_ceilings: Boolean to note whether interior ceiling
            Faces should be excluded from the resulting string. (Default: False).
        equest_version: An optional text string to denote the version of eQuest
            for which the INP definition will be generated. If unspecified
            or unrecognized, the latest version of eQuest will be used.

    Usage:

    .. code-block:: python

        import os
        from ladybug.futil import write_to_file
        from honeybee.model import Model
        from honeybee.room import Room
        from honeybee.config import folders

        # Crate an input Model
        room = Room.from_box('Tiny House Zone', 5, 10, 3)
        room.properties.energy.program_type = office_program
        room.properties.energy.add_default_ideal_air()
        model = Model('Tiny House', [room])

        # create the INP string for the model
        inp_str = model.to.inp(model)

        # write the final string into an INP
        inp = os.path.join(folders.default_simulation_folder, 'test_file', 'in.inp')
        write_to_file(inp, inp_str, True)
    """
    # duplicate model to avoid mutating it as we edit it for INP export
    original_model = model
    model = model.duplicate()
    # scale the model if the units are not feet
    if model.units != 'Feet':
        model.convert_to_units('Feet')
    # remove degenerate geometry within native DOE-2 tolerance
    try:
        model.remove_degenerate_geometry(DOE2_TOLERANCE)
    except ValueError:
        error = 'Failed to remove degenerate Rooms.\nYour Model units system is: {}. ' \
            'Is this correct?'.format(original_model.units)
        raise ValueError(error)
    # remove interior sub-faces as they cannot be placed in INP extrusion walls
    for face in model.faces:
        if isinstance(face.boundary_condition, Surface) and face.has_sub_faces:
            face.remove_sub_faces()
    # convert all of the Aperture geometries to rectangles so they can be translated
    model.rectangularize_apertures(
        subdivision_distance=RECT_WIN_SUBD, max_separation=0.0,
        merge_all=True, resolve_adjacency=False
    )
    # reset identifiers to valid DOE-2 U-Names that are derived from the display names
    for room in model.rooms:
        base_name = clean_doe2_string(room.display_name, GEO_CHARS - 2)
        room.display_name = clean_string(base_name)
        for face in room.faces:
            base_name = clean_doe2_string(face.display_name, GEO_CHARS - 2)
            face.display_name = clean_string(base_name)
            for ap in face.apertures:
                base_name = clean_doe2_string(ap.display_name, GEO_CHARS - 2)
                ap.display_name = clean_string(base_name)
            for dr in face.doors:
                base_name = clean_doe2_string(dr.display_name, GEO_CHARS - 2)
                dr.display_name = clean_string(base_name)
    for shade in model.shades:
        base_name = clean_doe2_string(shade.display_name, GEO_CHARS - 2)
        shade.display_name = clean_string(base_name)
    for shd_mesh in model.shade_meshes:
        base_name = clean_doe2_string(shd_mesh.display_name, GEO_CHARS - 2)
        shd_mesh.display_name = clean_string(base_name)
    model.reset_ids()
    # assign any doe2 properties previously supported through user_data
    for room in model.rooms:
        room.properties.doe2.apply_properties_from_user_data()

    # write the simulation parameters into the string
    model_str = ['INPUT ..\n\n']
    sim_par = simulation_par if simulation_par is not None else SimulationPar()
    model_str.append(sim_par.to_inp())

    # write all of the schedules
    all_day_scheds, all_week_scheds, all_year_scheds = [], [], []
    used_day_sched_ids, used_day_count = {}, 1
    all_scheds = model.properties.energy.schedules
    for sched in all_scheds:
        if isinstance(sched, ScheduleRuleset):
            year_schedule, week_schedules = sched.to_inp()
            # check that day schedules aren't referenced by other model schedules
            day_scheds = []
            for day in sched.day_schedules:
                sch_doe2_id = clean_doe2_string(day.identifier, RES_CHARS)
                if sch_doe2_id not in used_day_sched_ids:
                    day_scheds.append(day.to_inp(sched.schedule_type_limit))
                    used_day_sched_ids[sch_doe2_id] = day
                elif day != used_day_sched_ids[sch_doe2_id]:
                    new_day = day.duplicate()
                    new_day.identifier = 'Schedule Day {}'.format(used_day_count)
                    day_scheds.append(new_day.to_inp(sched.schedule_type_limit))
                    for i, week_sch in enumerate(week_schedules):
                        old_day_id = clean_doe2_string(day.identifier, RES_CHARS)
                        new_day_id = clean_doe2_string(new_day.identifier, RES_CHARS)
                        week_schedules[i] = week_sch.replace(old_day_id, new_day_id)
                    used_day_count += 1
            all_day_scheds.extend(day_scheds)
            all_week_scheds.extend(week_schedules)
            all_year_scheds.append(year_schedule)
        else:  # ScheduleFixedInterval
            year_schedule, week_schedules, year_schedule = sched.to_inp()
            all_day_scheds.extend(day_scheds)
            all_week_scheds.extend(week_schedules)
            all_year_scheds.append(year_schedule)
    model_str.append(header_comment_minor('Day Schedules'))
    model_str.extend(all_day_scheds)
    model_str.append(header_comment_minor('Week Schedules'))
    model_str.extend(all_week_scheds)
    model_str.append(header_comment_minor('Annual Schedules'))
    model_str.extend(all_year_scheds)

    # write all of the materials and constructions
    window_constructions = model.properties.energy.aperture_constructions()
    door_constructions = model.properties.energy.door_constructions()
    drc_ids = set([con.identifier for con in door_constructions])
    materials = []
    construction_strs = []
    all_constrs = model.properties.energy.constructions + \
        generic_construction_set.constructions_unique
    for constr in set(all_constrs):
        if isinstance(constr, OpaqueConstruction) and constr.identifier not in drc_ids:
            materials.extend(constr.materials)
            construction_strs.append(opaque_construction_to_inp(constr))
        elif isinstance(constr, AirBoundaryConstruction):
            construction_strs.append(air_construction_to_inp(constr))
    model_str.append(header_comment_minor('Materials / Layers / Constructions'))
    model_str.extend([opaque_material_to_inp(mat) for mat in set(materials)])
    model_str.extend(construction_strs)
    model_str.append(header_comment_minor('Glass Types'))
    for w_con in window_constructions:
        model_str.append(window_construction_to_inp(w_con))
    model_str.append(header_comment_minor('Door Construction'))
    for dr_con in door_constructions:
        if not isinstance(dr_con, OpaqueConstruction):
            dr_con = dr_con.duplicate()
            dr_con.identifier = dr_con.identifier + '_d'
        model_str.append(door_construction_to_inp(dr_con))

    # gather together all of the program types in a dictionary for switch statements
    switch_dict = {}
    for program in model.properties.energy.program_types:
        program_type_to_inp(program, switch_dict)

    # loop through rooms grouped by floor level and boundary to get polygons
    level_room_groups, level_geos, level_names = \
        group_rooms_by_doe2_level(model.rooms, model.tolerance)
    bldg_polygons, bldg_geo_defs = [], []
    for flr_rooms, flr_geo, flr_name in zip(level_room_groups, level_geos, level_names):
        # create the story definition
        rooms_f2c = [room.max.z - room.min.z for room in flr_rooms]
        sotry_f2f = max(rooms_f2c)
        median_room_f2c = sorted(rooms_f2c)[int(len(rooms_f2c) / 2)]
        if flr_geo is None:  # write the level with NO-SHAPE
            msg = 'Using NO-SHAPE for FLOOR "{}".'.format(flr_name)
            print(msg)
            flr_origin, _ = bounding_box([room.min for room in flr_rooms])
            flr_area = sum(room.floor_area for room in flr_rooms)
            flr_volume = sum(room.volume for room in flr_rooms)
            flr_keys = ['SHAPE', 'AREA', 'VOLUME', 'AZIMUTH', 'X', 'Y', 'Z',
                        'SPACE-HEIGHT', 'FLOOR-HEIGHT']
            flr_vals = ['NO-SHAPE', round(flr_area, GEO_DEC_COUNT),
                        round(flr_volume, GEO_DEC_COUNT), 0,
                        flr_origin.x, flr_origin.y, flr_origin.z,
                        round(median_room_f2c, 3), round(sotry_f2f, 3)]
        else:  # write the level with a POLYGON
            flr_polygon, pos_info = face_3d_to_inp(flr_geo, flr_name)
            flr_origin, _, _ = pos_info
            flr_keys = ['SHAPE', 'POLYGON', 'AZIMUTH', 'X', 'Y', 'Z',
                        'SPACE-HEIGHT', 'FLOOR-HEIGHT']
            flr_vals = ['POLYGON', '"{} Plg"'.format(flr_name), 0,
                        flr_origin.x, flr_origin.y, flr_origin.z,
                        round(median_room_f2c, 3), round(sotry_f2f, 3)]
            bldg_polygons.append(flr_polygon)
        r_mult = flr_rooms[0].multiplier
        if r_mult != 1 and all(room.multiplier == r_mult for room in flr_rooms):
            # set the multiplier for the entire story instead of room-by-room
            flr_keys.append('MULTIPLIER')
            flr_vals.append(r_mult)
            for room in flr_rooms:
                room.multiplier = 1
        flr_def = generate_inp_string(flr_name, 'FLOOR', flr_keys, flr_vals)
        bldg_geo_defs.append(flr_def)
        # add the room and face definitions + polygons
        for room in flr_rooms:
            room_polygons, room_defs = room_to_inp(
                room, flr_origin, median_room_f2c,
                exclude_interior_walls, exclude_interior_ceilings)
            bldg_polygons.extend(room_polygons)
            bldg_geo_defs.extend(room_defs)

    # loop through the shades and get their definitions and polygons
    shade_polygons, shade_geo_defs = [], []
    for shade in model.shades:
        shade_polygon, shade_def = shade_to_inp(shade, equest_version)
        if shade_polygon != '':  # shade written with a RECTANGLE
            shade_polygons.append(shade_polygon)
        shade_geo_defs.append(shade_def)
    for shade in model.shade_meshes:
        shade_polygon, shade_def = shade_mesh_to_inp(shade, equest_version)
        shade_polygons.extend(shade_polygon)
        shade_geo_defs.extend(shade_def)

    # write the building and shade geometry into the INP
    model_str.append(header_comment_minor('Polygons'))
    model_str.extend(bldg_polygons)
    model_str.append(header_comment_minor('Wall Parameters'))
    model_str.append(header_comment_minor('Fixed and Building Shades'))
    model_str.extend(shade_polygons)
    model_str.extend(shade_geo_defs)
    model_str.append(header_comment_minor('Misc Cost Related Objects'))
    model_str.append(header_comment_major('Performance Curves'))
    model_str.append(header_comment_major('Floors / Spaces / Walls / Windows / Doors'))
    model_str.append(switch_dict_to_space_inp(switch_dict))
    model_str.extend(bldg_geo_defs)

    # write in placeholder headers for various HVAC components
    model_str.append(header_comment_major('Electric & Fuel Meters'))
    for meter in ('Electric Meters', 'Fuel Meters', 'Master Meters'):
        model_str.append(header_comment_minor(meter))
    model_str.append(header_comment_major('HVAC Circulation Loops / Plant Equipment'))
    hvac_comp_types = (
        'Pumps', 'Heat Exchangers', 'Circulation Loops', 'Chillers', 'Boilers',
        'Domestic Water Heaters', 'Heat Rejection', 'Tower Free Cooling',
        'Photovoltaic Modules', 'Electric Generators', 'Thermal Storage',
        'Ground Loop Heat Exchangers', 'Compliance DHW (residential dwelling units)')
    for comp in hvac_comp_types:
        model_str.append(header_comment_minor(comp))
    model_str.append(header_comment_major('Steam & Chilled Water Meters'))
    model_str.append(header_comment_minor('Steam Meters'))
    model_str.append(header_comment_minor('Chilled Water Meters'))
    model_str.append(header_comment_major('HVAC Systems / Zones'))
    model_str.append(switch_dict_to_zone_inp(switch_dict))

    # assign HVAC systems given the specified hvac_mapping
    if hvac_mapping.upper() == 'STORY':
        hvac_rooms = level_room_groups
        hvac_names = ['{}_Sys'.format(name) for name in level_names]
    else:
        hvac_rooms, hvac_names = group_rooms_by_doe2_hvac(model, hvac_mapping)
    for hvac_name, rooms in zip(hvac_names, hvac_rooms):
        # create the definition of the HVAC
        hvac_keys = ('TYPE', 'HEAT-SOURCE', 'SYSTEM-REPORTS')
        hvac_vals = ('SUM', 'NONE', 'NO')
        hvac_def = generate_inp_string(hvac_name, 'SYSTEM', hvac_keys, hvac_vals)
        model_str.append(hvac_def)
        for room in rooms:
            space_name = clean_doe2_string(room.identifier, GEO_CHARS)
            zone_name = '{}_Zn'.format(space_name)
            zone_type = room_doe2_conditioning_type(room)
            zone_keys = ['TYPE', 'SIZING-OPTION', 'SPACE']
            zone_vals = [zone_type, 'ADJUST-LOADS', '"{}"'.format(space_name)]
            if room.properties.energy.is_conditioned:
                if room.properties.energy._setpoint is not None:
                    stp_kwd, stp_val = setpoint_to_inp(room.properties.energy._setpoint)
                    zone_keys.extend(stp_kwd)
                    zone_vals.extend(stp_val)
                vt_kwd, vt_val = ventilation_to_inp(room.properties.energy._ventilation)
                zone_keys.extend(vt_kwd)
                zone_vals.extend(vt_val)
                hvac_kwd, hvac_val = room.properties.doe2.to_inp()
                zone_keys.extend(hvac_kwd)
                zone_vals.extend(hvac_val)
            zone_def = generate_inp_string(zone_name, 'ZONE', zone_keys, zone_vals)
            model_str.append(zone_def)

    # provide a few last comment headers and end the file
    model_str.append(header_comment_major('Metering & Misc HVAC'))
    model_str.append(header_comment_minor('Equipment Controls'))
    model_str.append(header_comment_minor('Load Management'))
    model_str.append(header_comment_major('Utility Rates'))
    for rate in ('Ratchets', 'Block Charges', 'Utility Rates'):
        model_str.append(header_comment_minor(rate))
    model_str.append(header_comment_major('Output Reporting'))
    report_types = (
        'Loads Non-Hourly Reporting', 'Systems Non-Hourly Reporting',
        'Plant Non-Hourly Reporting', 'Economics Non-Hourly Reporting',
        'Hourly Reporting', 'THE END')
    for report in report_types:
        model_str.append(header_comment_minor(report))
    model_str.append('END ..\nCOMPUTE ..\nSTOP ..\n')

    # create the final string and ensure that it is windows-compatible
    inp_str = '\n'.join(model_str)
    if os.name != 'nt':  # we are on a unix-based system
        inp_str = inp_str.replace('\n', '\r\n')
    return inp_str


def room_doe2_conditioning_type(room):
    """Get the DOE-2 conditioning type to be assigned to both the Space and Zone.

    Args:
        room: A Honeybee Room for which the conditioning type will be returned.
    """
    if room.exclude_floor_area:
        return 'PLENUM'
    elif room.properties.energy.is_conditioned:
        return 'CONDITIONED'
    else:
        return 'UNCONDITIONED'
