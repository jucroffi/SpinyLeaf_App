"""Method to translate a Room to a VisualizationSet."""
from ladybug_display.visualization import VisualizationSet, ContextGeometry

from ._util import _process_wireframe
from .face import _face_display_geometry, _add_display_shade


def room_to_vis_set(room, color_by='type'):
    """Translate a Honeybee Room to a VisualizationSet.

    Args:
        room: A Honeybee Room object to be converted to a VisualizationSet.
        color_by: Text for the property that dictates the colors of the Room
            geometry. (Default: type). Choose from the following:

            * type
            * boundary_condition

    Returns:
        A VisualizationSet object that represents the Room with a single ContextGeometry.
    """
    # get the basic properties for geometry conversion
    color_by_attr = 'type_color' if color_by.lower() == 'type' else 'bc_color'
    # convert all geometry into DisplayFace3D
    dis_geos = []
    for face in room.faces:
        dis_geos.extend(_face_display_geometry(face, color_by_attr))
    _add_display_shade(room, dis_geos, color_by_attr)
    # build the VisualizationSet and ContextGeometry
    con_geo = ContextGeometry(room.identifier, dis_geos)
    con_geo.display_name = room.display_name
    vis_set = VisualizationSet(room.identifier, [con_geo])
    vis_set.display_name = room.display_name
    return vis_set


def room_to_vis_set_wireframe(
        room, include_sub_faces=True, include_shades=True, color=None):
    """Get a VisualizationSet with a single ContextGeometry for the room wireframe.

    Args:
        room: A Honeybee Room object to be translated to a wireframe.
        include_sub_faces: Boolean for whether the wireframe should include sub-faces
            of the Room. (Default: True).
        include_shades: Boolean for whether the wireframe should include shades
            of the Room. (Default: True).
        color: An optional Color object to set the color of the wireframe.
            If None, the color will be black.

    Returns:
        A VisualizationSet with a single ContextGeometry and a list of
        DisplayLineSegment3D for the wireframe of the Room.
    """
    wireframe = []
    for face in room.faces:
        _process_wireframe(face.geometry, wireframe, color, 2)
        if include_sub_faces:
            for ap in face._apertures:
                _process_wireframe(ap.geometry, wireframe, color)
                if include_shades:
                    for shd in ap.shades:
                        _process_wireframe(shd.geometry, wireframe, color)
            for dr in face._doors:
                _process_wireframe(dr.geometry, wireframe, color)
                if include_shades:
                    for shd in dr.shades:
                        _process_wireframe(shd.geometry, wireframe, color)
            if include_shades:
                for shd in face.shades:
                    _process_wireframe(shd.geometry, wireframe, color)
    if include_shades:
        for shd in room.shades:
            _process_wireframe(shd.geometry, wireframe, color)

    vis_set = VisualizationSet(
        room.identifier, [ContextGeometry('Wireframe', wireframe)])
    vis_set.display_name = room.display_name
    return vis_set
