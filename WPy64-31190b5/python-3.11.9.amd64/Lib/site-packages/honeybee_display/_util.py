"""Utility functions shared by all of the modules."""
from ladybug_display.geometry3d import DisplayLineSegment3D


def _process_wireframe(face3d, wireframe, color=None, line_width=1):
    """Process the boundary and holes into DisplayLinesegment3D."""
    for seg in face3d.boundary_segments:
        wireframe.append(DisplayLineSegment3D(seg, color, line_width))
    if face3d.has_holes:
        for hole in face3d.hole_segments:
            for seg in hole:
                wireframe.append(DisplayLineSegment3D(seg, color, line_width))
