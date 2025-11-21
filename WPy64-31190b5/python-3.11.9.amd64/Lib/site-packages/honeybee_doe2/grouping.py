# coding=utf-8
"""Methods for grouping rooms to comply with INP rules."""
from __future__ import division
import math

from ladybug_geometry.geometry2d import Point2D, Polygon2D
from ladybug_geometry.geometry3d import Vector3D, Point3D, Face3D
from honeybee.typing import clean_doe2_string
from honeybee.room import Room

from .config import DOE2_TOLERANCE, FLOOR_LEVEL_TOL, RES_CHARS


def group_rooms_by_doe2_level(rooms, model_tolerance):
    """Group Honeybee Rooms according to acceptable floor levels in DOE-2.

    This means that not only will Rooms be on separate DOE-2 levels if their floor
    heights differ but also Rooms that share the same floor height but are
    disconnected from one another in plan will also be separate levels.
    For example, when the model is of two towers on a plinth, each tower will
    get its own separate level group.

    Args:
        rooms: A list of Honeybee Rooms to be grouped.
        model_tolerance: The tolerance of the model that the Rooms originated from.

    Returns:
        A tuple with three elements.

        -   room_groups: A list of lists where each sub-list contains Honeybee
            Rooms that should be on the same DOE-2 level.

        -   level_geometries: A list of Face3D with the same length as the
            room_groups, which contains the geometry that represents each floor
            level. These geometries will always be pointing upwards so that
            their vertices are counter-clockwise when viewed from above. They
            will also have colinear vertices removed such that they are ready
            to be translated to INP POLYGONS.

        -   level_names: A list of text strings that align with the level
            geometry and contain suggested names for the DOE-2 levels.
    """
    # set up lists of the outputs to be populated
    room_groups, level_geometries, level_names, existing_levels = [], [], [], {}

    # first group the rooms by floor height
    grouped_rooms, _ = Room.group_by_floor_height(rooms, FLOOR_LEVEL_TOL)
    for fi, room_group in enumerate(grouped_rooms):
        # determine a base name for the level using the story assigned to the rooms
        level_name = clean_doe2_string(room_group[0].story, RES_CHARS - 8) \
            if room_group[0].story is not None else 'Level_{}'.format(fi)
        if level_name in existing_levels:
            existing_levels[level_name] += 1
            level_name = level_name + str(existing_levels[level_name])
        else:
            existing_levels[level_name] = 1

        # then, group the rooms by contiguous horizontal boundary
        floor_geos = []
        for room in room_group:
            if room.properties.doe2.space_polygon_geometry is not None:
                floor_geos.append(room.properties.doe2.space_polygon_geometry)
            else:
                try:
                    flr_geo = room.horizontal_floor_boundaries(tolerance=model_tolerance)
                    if len(flr_geo) == 0:  # possible when Rooms have no floors
                        flr_geo = room.horizontal_boundary(tolerance=model_tolerance)
                        floor_geos.append(flr_geo)
                    else:
                        floor_geos.extend(flr_geo)
                except Exception:  # level geometry is overlapping or not clean
                    pass

        # join all of the floors into horizontal boundaries
        hor_bounds = _grouped_floor_boundary(floor_geos, model_tolerance)

        # if we got lucky and everything is one contiguous polygon, we're done!
        if len(hor_bounds) == 0:  # we will write the story with NO-SHAPE
            room_groups.append(room_group)
            level_geometries.append(None)
            level_names.append(level_name)
        elif len(hor_bounds) == 1:  # just one clean polygon for the level
            flr_geo = hor_bounds[0]
            flr_geo = flr_geo if flr_geo.normal.z >= 0 else flr_geo.flip()
            if flr_geo.has_holes:  # remove holes as we only care about the boundary
                flr_geo = Face3D(flr_geo.boundary, flr_geo.plane)
            flr_geo = flr_geo.remove_colinear_vertices(tolerance=DOE2_TOLERANCE)
            room_groups.append(room_group)
            level_geometries.append(flr_geo)
            level_names.append(level_name)
        else:  # we need to figure out which Room belongs to which geometry
            # first get a set of Point2Ds that are inside each room in plan
            room_pts, z_axis = [], Vector3D(0, 0, 1)
            for room in room_group:
                for face in room.faces:
                    if math.degrees(z_axis.angle(face.normal)) >= 91:
                        down_geo = face.geometry
                        break
                room_pt3d = down_geo.center if down_geo.is_convex else \
                    down_geo.pole_of_inaccessibility(DOE2_TOLERANCE)
                room_pts.append(Point2D(room_pt3d.x, room_pt3d.y))
            # loop through floor geometries and determine all rooms associated with them
            for si, flr_geo in enumerate(hor_bounds):
                flr_geo = flr_geo if flr_geo.normal.z >= 0 else flr_geo.flip()
                if flr_geo.has_holes:  # remove holes as we only care about the boundary
                    flr_geo = Face3D(flr_geo.boundary, flr_geo.plane)
                flr_geo = flr_geo.remove_colinear_vertices(tolerance=DOE2_TOLERANCE)
                flr_poly = Polygon2D([Point2D(pt.x, pt.y) for pt in flr_geo.boundary])
                flr_rooms = []
                for room, room_pt in zip(room_group, room_pts):
                    if flr_poly.is_point_inside_bound_rect(room_pt):
                        flr_rooms.append(room)
                room_groups.append(flr_rooms)
                level_geometries.append(flr_geo)
                level_names.append('{}_Section{}'.format(level_name, si))

    # return all of the outputs
    return room_groups, level_geometries, level_names


def group_rooms_by_doe2_hvac(model, hvac_mapping):
    """Group Honeybee Rooms according to HVAC logic.

    Args:
        model: A Honeybee Model for which Rooms will be grouped for HVAC assignment.
        hvac_mapping: Text to indicate how HVAC systems should be assigned.
            Model will use only one HVAC system for the whole model and
            AssignedHVAC will follow how the HVAC systems have been assigned
            to the Rooms.properties.energy.hvac. Choose from the options below.

            * Room
            * Model
            * AssignedHVAC

    Returns:
        A tuple with three elements.

        -   room_groups: A list of lists where each sub-list contains Honeybee
            Rooms that should ave the same HVAC system.

        -   hvac_names: A list of text strings that align with the room_groups
            and contain suggested names for the DOE-2 HVAC systems.
    """
    # clean up the hvac_mapping text
    hvac_mapping = hvac_mapping.upper().replace('-', '').replace(' ', '')

    # determine the mapping to be used
    if hvac_mapping == 'MODEL':
        hvac_name = clean_doe2_string('{}_Sys'.format(model.display_name), RES_CHARS)
        return [model.rooms], [hvac_name]
    elif hvac_mapping == 'ROOM':
        hvac_names = [clean_doe2_string('{}_Sys'.format(room.identifier), RES_CHARS)
                      for room in model.rooms]
        room_groups = [[room] for room in model.rooms]
    else:  # assume that it is the assigned HVAC
        hvac_dict = {}
        for room in model.rooms:
            if room.properties.energy.hvac is not None:
                hvac_id = room.properties.energy.hvac.display_name
                try:
                    hvac_dict[hvac_id].append(room)
                except KeyError:  # the first time that we are encountering the HVAC
                    hvac_dict[hvac_id] = [room]
            else:
                try:
                    hvac_dict['Unassigned'].append(room)
                except KeyError:  # the first time that we have an unassigned room
                    hvac_dict['Unassigned'] = [room]
        room_groups, hvac_names, existing_dict = [], [], {}
        for hvac_name, rooms in hvac_dict.items():
            room_groups.append(rooms)
            hvac_doe2_name = clean_doe2_string(hvac_name, RES_CHARS - 2)
            if hvac_doe2_name in existing_dict:
                existing_dict[hvac_doe2_name] += 1
                hvac_names.append(hvac_doe2_name + str(existing_dict[hvac_doe2_name]))
            else:
                existing_dict[hvac_doe2_name] = 1
                hvac_names.append(hvac_doe2_name)

    return room_groups, hvac_names


def _grouped_floor_boundary(floor_geos, tolerance=0.01):
    """Get a list of Face3D for the boundary around several horizontal Face3Ds.

    Args:
        floor_geos: A list of Honeybee Rooms for which the horizontal boundary will
            be computed.
        tolerance: The maximum difference between coordinate values of two
            vertices at which they can be considered equivalent. (Default: 0.01,
            suitable for objects in meters).
    """
    # remove colinear vertices and degenerate faces
    clean_floor_geos = []
    for geo in floor_geos:
        try:
            clean_floor_geos.append(geo.remove_colinear_vertices(tolerance))
        except AssertionError:  # degenerate geometry to ignore
            pass
    if len(clean_floor_geos) == 0:
        return []  # no Room boundary to be found

    # convert the floor Face3Ds into counterclockwise Polygon2Ds
    floor_polys, z_vals = [], []
    for flr_geo in clean_floor_geos:
        z_vals.append(flr_geo.min.z)
        b_poly = Polygon2D([Point2D(pt.x, pt.y) for pt in flr_geo.boundary])
        floor_polys.append(b_poly)
        if flr_geo.has_holes:
            for hole in flr_geo.holes:
                h_poly = Polygon2D([Point2D(pt.x, pt.y) for pt in hole])
                floor_polys.append(h_poly)
    z_min = min(z_vals)

    # find the joined intersected boundary
    closed_polys = Polygon2D.joined_intersected_boundary(floor_polys, tolerance)

    # remove colinear vertices from the resulting polygons
    clean_polys = []
    for poly in closed_polys:
        try:
            clean_polys.append(poly.remove_colinear_vertices(tolerance))
        except AssertionError:
            pass  # degenerate polygon to ignore

    # figure out if polygons represent holes in the others and make Face3D
    if len(clean_polys) == 0:
        return []
    elif len(clean_polys) == 1:  # can be represented with a single Face3D
        pts3d = [Point3D(pt.x, pt.y, z_min) for pt in clean_polys[0]]
        return [Face3D(pts3d)]
    else:  # need to separate holes from distinct Face3Ds
        bound_faces = []
        for poly in clean_polys:
            pts3d = tuple(Point3D(pt.x, pt.y, z_min) for pt in poly)
            bound_faces.append(Face3D(pts3d))
        return Face3D.merge_faces_to_holes(bound_faces, tolerance)
