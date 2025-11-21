"""Handlers to convert north vectors to angles."""
import math

from ladybug_geometry.geometry2d.pointvector import Vector2D


def north_vector_to_angle(value):
    """Translate a north vector into a north angle in degrees.

        Args:
            value: Either a Vector2D for the north direction or a number between
                -360 and 360 for the counterclockwise difference between the North
                and the positive Y-axis in degrees.

        Returns:
            float -- north angle in degrees.
    """
    if hasattr(value, 'X') and hasattr(value, 'Y'):
        vec = Vector2D(value.X, value.Y)
        north = math.degrees(vec.angle_clockwise(Vector2D(0, 1)))
    elif isinstance(value, Vector2D):
        north = math.degrees(value.angle_clockwise(Vector2D(0, 1)))
    else:
        try:
            north = float(value)
        except (ValueError, TypeError):
            raise ValueError(
                'north input should be a number or a vector. Not {}.'.format(type(value))
            )
    return north
