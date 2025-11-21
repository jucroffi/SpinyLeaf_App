"""Method to translate a Door to a VisualizationSet."""
from ladybug_display.geometry3d import DisplayFace3D
from ladybug_display.visualization import VisualizationSet, ContextGeometry

from ._util import _process_wireframe


def door_to_vis_set(door, color_by='type'):
    """Translate a Honeybee Door to a VisualizationSet.

    Args:
        door: A Honeybee Door object to be converted to a VisualizationSet.
        color_by: Text for the property that dictates the colors of the Door
            geometry. (Default: type). Choose from the following:

            * type
            * boundary_condition

    Returns:
        A VisualizationSet object that represents the Door with a single ContextGeometry.
    """
    # get the basic properties for geometry conversion
    color_by_attr = 'type_color' if color_by.lower() == 'type' else 'bc_color'
    d_mod = 'SurfaceWithEdges'
    # convert all geometry into DisplayFace3D
    dis_geos = []
    a_col = getattr(door, color_by_attr)
    dis_geos.append(DisplayFace3D(door.geometry, a_col, d_mod))
    for shd in door.shades:
        s_col = getattr(shd, color_by_attr)
        dis_geos.append(DisplayFace3D(shd.geometry, s_col, d_mod))
    # build the VisualizationSet and ContextGeometry
    con_geo = ContextGeometry(door.identifier, dis_geos)
    con_geo.display_name = door.display_name
    vis_set = VisualizationSet(door.identifier, [con_geo])
    vis_set.display_name = door.display_name
    return vis_set


def door_to_vis_set_wireframe(door, include_shades=True, color=None):
    """Get a VisualizationSet with a single ContextGeometry for the door wireframe.

    Args:
        door: A Honeybee Door object to be translated to a wireframe.
        include_shades: Boolean for whether the wireframe should include shades
            of the Door. (Default: True).
        color: An optional Color object to set the color of the wireframe.
            If None, the color will be black.

    Returns:
        A VisualizationSet with a single ContextGeometry and a list of
        DisplayLineSegment3D for the wireframe of the Door.
    """
    wireframe = []
    _process_wireframe(door.geometry, wireframe, color)
    if include_shades:
        for shd in door.shades:
            _process_wireframe(shd.geometry, wireframe, color)

    vis_set = VisualizationSet(
        door.identifier, [ContextGeometry('Wireframe', wireframe)])
    vis_set.display_name = door.display_name
    return vis_set
