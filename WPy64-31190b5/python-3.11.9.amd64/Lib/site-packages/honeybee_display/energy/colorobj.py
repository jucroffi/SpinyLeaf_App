"""Method to translate a Color Room/Face objects to a VisualizationSet."""
import math

from ladybug_geometry.geometry3d import Vector3D, Point3D, Polyline3D, Plane, \
    Face3D, Polyface3D
from ladybug_display.geometry3d import DisplayText3D
from ladybug_display.visualization import VisualizationSet, ContextGeometry, \
    AnalysisGeometry, VisualizationData
from honeybee.units import conversion_factor_to_meters, parse_distance_string
from honeybee.facetype import Floor

from ..colorobj import _room_wireframe, _process_wireframe


def energy_color_room_to_vis_set(
        color_room, include_wireframe=True, text_labels=False,
        units=None, tolerance=0.01):
    """Translate a Honeybee-Energy ColorRoom to a VisualizationSet.

    Args:
        color_room: A Honeybee-Energy ColorRoom object to be converted to a
            VisualizationSet.
        include_wireframe: Boolean to note whether a ContextGeometry just for
            the Wireframe (in LineSegment3D) should be included. (Default: True).
        text_labels: A boolean to note whether the results should be expressed
            as a colored AnalysisGeometry (False) or a ContextGeometry as text
            labels (True). (Default: False).
        units: Optional text, which will be used to set the default maximum text
            height and the distance of the text to the ground. If None, some
            generic defaults will be used. (Default: None).
        tolerance: Optional tolerance value, which is used to compute the text
            label point for concave geometries. (Default: 0.01).

    Returns:
        A VisualizationSet object that represents the ColorRoom with an
        AnalysisGeometry when text_labels is False or a ContextGeometry when
        text_labels is True. It can also optionally include a ContextGeometry
        for the wireframe.
    """
    # set up an empty visualization set
    vis_set = VisualizationSet(color_room.data_type_text.replace(' ', '_'), ())
    vis_set.display_name = color_room.data_type_text

    # use text labels if requested
    if text_labels:
        txt_height, font, f_str = _process_leg_par_for_text(color_room)
        # loop through the rooms and create the text labels
        max_txt_h, p_tol = float('inf'), 0.01
        if units is not None:
            fac_to_m = conversion_factor_to_meters(units)
            max_txt_h = 0.25 / fac_to_m
            max_txt_v = 1.0 / fac_to_m
            p_tol = parse_distance_string('0.01m', units)
        label_text = []
        for room_val, room in zip(color_room.matched_values, color_room.matched_rooms):
            room_prop = f_str % room_val
            # compute the center point for the text
            if units is not None:
                room_h = room.geometry.max.z - room.geometry.min.z
                m_vec = Vector3D(0, 0, max_txt_v) if room_h > max_txt_v * 2 \
                    else Vector3D(0, 0, room_h / 2)
                floor_faces = [face.geometry for face in room.faces
                               if isinstance(face.type, Floor)]
                if len(floor_faces) == 1:
                    flr_geo = floor_faces[0]
                    base_pt = flr_geo.center if flr_geo.is_convex else \
                        flr_geo.pole_of_inaccessibility(p_tol)
                elif len(floor_faces) == 0:
                    c_pt = room.geometry.center
                    base_pt = Point3D(c_pt.x, c_pt.y, room.geometry.min.z)
                else:
                    floor_p_face = Polyface3D.from_faces(floor_faces, tolerance)
                    ne = floor_p_face.naked_edges
                    floor_outline = Polyline3D.join_segments(ne, tolerance)[0]
                    flr_geo = Face3D(floor_outline.vertices[:-1])
                    base_pt = flr_geo.center if flr_geo.is_convex else \
                        flr_geo.pole_of_inaccessibility(p_tol)
                base_pt = base_pt.move(m_vec)
                base_plane = Plane(Vector3D(0, 0, 1), base_pt)
            else:
                base_pt = room.geometry.center
            base_plane = Plane(Vector3D(0, 0, 1), base_pt)
            # get the text height
            if txt_height is None:  # auto-calculate default text height
                txt_len = len(room_prop) if len(room_prop) > 10 else 10
                txt_h = (room.geometry.max.x - room.geometry.min.x) / txt_len
            else:
                txt_h = txt_height
            txt_h = max_txt_h if txt_h > max_txt_h else txt_h
            # create the text label
            label = DisplayText3D(
                room_prop, base_plane, txt_h, font=font,
                horizontal_alignment='Center', vertical_alignment='Middle')
            label_text.append(label)  # append everything to the list
        con_geo = ContextGeometry(vis_set.identifier, label_text)
        con_geo.display_name = vis_set.display_name
        vis_set.add_geometry(con_geo)
    else:  # use a colored AnalysisGeometry
        # create the analysis geometry
        vis_data = VisualizationData(
            color_room.matched_values, color_room.legend_parameters,
            color_room.data_type, str(color_room.unit))
        geo = tuple(room.geometry for room in color_room.matched_rooms)
        a_geo = AnalysisGeometry(vis_set.identifier, geo, [vis_data])
        a_geo.display_name = vis_set.display_name
        vis_set.add_geometry(a_geo)

    # loop through all of the rooms and add their wire frames
    if include_wireframe:
        vis_set.add_geometry(_room_wireframe(color_room.rooms))
    return vis_set


def color_face_to_vis_set(
        color_face, include_wireframe=True, text_labels=False,
        units=None, tolerance=0.01):
    """Translate a Honeybee ColorFace to a VisualizationSet.

    Args:
        color_face: A Honeybee ColorFace object to be converted to a VisualizationSet.
        include_wireframe: Boolean to note whether a ContextGeometry just for
            the Wireframe (in LineSegment3D) should be included. (Default: True).
        text_labels: A boolean to note whether the attribute assigned to the
            ColorFace should be expressed as a colored AnalysisGeometry (False)
            or a ContextGeometry as text labels (True). (Default: False).
        units: Optional text, which will be used to set the default maximum text
            height and the distance of the text to the ground. If None, some
            generic defaults will be used. (Default: None).
        tolerance: Tolerance value, which is used to eliminate very small
            text. (Default: 0.01).

    Returns:
        A VisualizationSet object that represents the ColorFace with an
        AnalysisGeometry when text_labels is False or a ContextGeometry when
        text_labels is True. It can also optionally include a ContextGeometry
        for the wireframe.
    """
    # set up an empty visualization set
    vis_set = VisualizationSet(color_face.data_type_text.replace(' ', '_'), ())
    vis_set.display_name = color_face.data_type_text

    # use text labels if requested
    if text_labels:
        # set up default variables
        max_txt_h, p_tol = float('inf'), 0.01
        if units is not None:
            fac_to_m = conversion_factor_to_meters(units)
            max_txt_h = 0.25 / fac_to_m
            p_tol = parse_distance_string('0.01m', units)
        txt_height, font, f_str = _process_leg_par_for_text(color_face)
        # loop through the faces and create the text labels
        label_text = []
        face_zip_obj = zip(color_face.matched_values, color_face.matched_flat_geometry)
        for face_val, f_geo in face_zip_obj:
            cent_pt = f_geo.center if f_geo.is_convex else \
                f_geo.pole_of_inaccessibility(p_tol)
            base_plane = Plane(f_geo.normal, cent_pt)
            face_prop = f_str % face_val
            if base_plane.y.z < 0:  # base plane pointing downwards; rotate it
                base_plane = base_plane.rotate(base_plane.n, math.pi, base_plane.o)
            if txt_height is None:  # auto-calculate default text height
                txt_len = len(face_prop) if len(face_prop) > 10 else 10
                dims = [
                    (f_geo.max.x - f_geo.min.x),
                    (f_geo.max.y - f_geo.min.y),
                    (f_geo.max.z - f_geo.min.z)]
                dims.sort()
                txt_h = dims[1] / txt_len
            else:
                txt_h = txt_height
            if txt_h < tolerance:
                continue
            txt_h = max_txt_h if txt_h > max_txt_h else txt_h
            # move base plane origin a little to avoid overlaps of adjacent labels
            if base_plane.n.x != 0:
                m_vec = base_plane.y if base_plane.n.x < 0 else -base_plane.y
            else:
                m_vec = base_plane.y if base_plane.n.z < 0 else -base_plane.y
            base_plane = base_plane.move(m_vec * txt_h)
            # create the text label
            label = DisplayText3D(
                face_prop, base_plane, txt_h, font=font,
                horizontal_alignment='Center', vertical_alignment='Middle')
            label_text.append(label)  # append everything to the list
        con_geo = ContextGeometry(vis_set.identifier, label_text)
        con_geo.display_name = vis_set.display_name
        vis_set.add_geometry(con_geo)
    else:  # use a colored AnalysisGeometry
        # create the analysis geometry
        vis_data = VisualizationData(
            color_face.matched_values, color_face.legend_parameters,
            color_face.data_type, str(color_face.unit))
        a_geo = AnalysisGeometry(
            vis_set.identifier, color_face.matched_flat_geometry, [vis_data])
        a_geo.display_name = vis_set.display_name
        vis_set.add_geometry(a_geo)

    # loop through all of the rooms and add their wire frames
    if include_wireframe:
        wireframe = []
        for face in color_face.faces:
            _process_wireframe(face, wireframe, 2)
            for ap in face._apertures:
                _process_wireframe(ap, wireframe)
            for dr in face._doors:
                _process_wireframe(dr, wireframe)
        con_geo = ContextGeometry('Wireframe', wireframe)
        vis_set.add_geometry(con_geo)
    return vis_set


def _process_leg_par_for_text(color_obj):
    """Get the relevant Legend Parameters for DisplayText3D."""
    txt_height = None if color_obj.legend_parameters.is_text_height_default \
        else color_obj.legend_parameters.text_height
    font = color_obj.legend_parameters.font
    f_str = '%.{}f'.format(color_obj.legend_parameters.decimal_count)
    return txt_height, font, f_str
