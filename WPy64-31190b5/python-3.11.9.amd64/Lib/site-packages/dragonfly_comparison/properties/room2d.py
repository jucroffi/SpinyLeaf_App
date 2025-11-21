# coding=utf-8
"""Room2D Comparison Properties."""
import math

from ladybug_geometry.geometry3d import Point3D, Vector3D, Plane, Face3D
from dragonfly.windowparameter import _WindowParameterBase
from dragonfly.skylightparameter import _SkylightParameterBase, DetailedSkylights
import dragonfly.windowparameter as glzpar
import dragonfly.skylightparameter as skypar


class Room2DComparisonProperties(object):
    """Comparison Properties for Dragonfly Room2D.

    Args:
        host: A dragonfly_core Room2D object that hosts these properties.
        comparison_floor_geometry: A single horizontal Face3D representing the
            floor plate of the Room2D to which the host Room2D is being
            compared. (Default: None).
        comparison_windows: A list of WindowParameter objects that dictate the
            window geometries of the Room2D to which the host Room2D is being
            compared. (Default: None).
        comparison_skylight: A SkylightParameters object that dictate the
            skylight geometries of the Room2D to which the host Room2D is being
            compared. (Default: None).

    Properties:
        * host
        * comparison_floor_geometry
        * comparison_windows
        * comparison_skylight
        * floor_area
        * floor_area_difference
        * floor_area_abs_difference
        * floor_area_percent_change
        * wall_sub_face_area
        * wall_sub_face_area_difference
        * wall_sub_face_area_abs_difference
        * wall_sub_face_area_percent_change
        * roof_sub_face_area
        * roof_sub_face_area_difference
        * roof_sub_face_area_abs_difference
        * roof_sub_face_area_percent_change
        * sub_face_area
        * sub_face_area_difference
        * sub_face_area_abs_difference
        * sub_face_area_percent_change
    """
    __slots__ = ('_host', '_comparison_floor_geometry', '_comparison_windows',
                 '_comparison_skylight')

    def __init__(self, host, comparison_floor_geometry=None, comparison_windows=None,
                 comparison_skylight=None):
        """Initialize Room2D Comparison properties."""
        self._host = host
        self.comparison_floor_geometry = comparison_floor_geometry
        self.comparison_windows = comparison_windows
        self.comparison_skylight = comparison_skylight

    @property
    def host(self):
        """Get the Room2D object hosting these properties."""
        return self._host

    @property
    def comparison_floor_geometry(self):
        """Get or set a horizontal Face3D for the Room2D to which the host is compared.

        If not set, all properties relating to floor geometry comparison will
        be zero (aka. unchanged).
        """
        return self._comparison_floor_geometry

    @comparison_floor_geometry.setter
    def comparison_floor_geometry(self, value):
        if value is not None:
            # process the floor_geometry
            assert isinstance(value, Face3D), \
                'Expected ladybug_geometry Face3D. Got {}'.format(type(value))
            if value.normal.z < 0:  # ensure upward-facing Face3D
                value = value.flip()
            # ensure a global 2D origin, which helps in solve adjacency and the dict schema
            o_pl = Plane(Vector3D(0, 0, 1), Point3D(0, 0, value.plane.o.z))
            value = Face3D(value.boundary, o_pl, value.holes)
        self._comparison_floor_geometry = value

    @property
    def comparison_windows(self):
        """Get or set a tuple of WindowParameters describing how to generate windows.

        If not set, all properties relating to wall sub-face geometry comparison
        will be zero (aka. unchanged).
        """
        return self._comparison_windows

    @comparison_windows.setter
    def comparison_windows(self, value):
        if value is not None:
            if not isinstance(value, tuple):
                value = tuple(value)
            for val in value:
                if val is not None:
                    assert isinstance(val, _WindowParameterBase), \
                        'Expected Window Parameters. Got {}'.format(type(value))
        self._comparison_windows = value

    @property
    def comparison_skylight(self):
        """Get or set SkylightParameters describing how to generate skylights.

        If not set, all properties relating to roof sub-face geometry comparison
        will be zero (aka. unchanged).
        """
        return self._comparison_skylight

    @comparison_skylight.setter
    def comparison_skylight(self, value):
        if value is not None:
            assert isinstance(value, _SkylightParameterBase), \
                'Expected Skylight Parameters. Got {}'.format(type(value))
        self._comparison_skylight = value

    @property
    def floor_segments(self):
        """Get a list of LineSegment3D objects for each wall of the comparison Room."""
        fg = self.comparison_floor_geometry
        if fg is None:
            return None
        return fg.boundary_segments if fg.holes is None else \
            fg.boundary_segments + tuple(s for hole in fg.hole_segments for s in hole)

    @property
    def floor_area(self):
        """Get a number for the floor area of the Room2D to which the host is compared.
        """
        if self.comparison_floor_geometry is None:
            return self.host.floor_area
        return self.comparison_floor_geometry.area

    @property
    def floor_area_difference(self):
        """Get a number for the difference between the host and comparison floor area.

        This number will be positive if the floor area increased in the host room
        compared to the comparison room and negative if it decreased.
        """
        return self.host.floor_area - self.floor_area

    @property
    def floor_area_abs_difference(self):
        """Get a number for the difference between the host and comparison floor area.
        """
        return abs(self.floor_area_difference)

    @property
    def floor_area_percent_change(self):
        """Get a number between 0 an 100 for the percent change between floor areas.
        """
        return (self.floor_area_abs_difference / self.floor_area) * 100

    @property
    def wall_sub_face_area(self):
        """Get a number for the wall sub-face area of the comparison Room2D.

        This includes both Apertures and Doors.
        """
        if self.comparison_windows is None or self.comparison_floor_geometry is None:
            return self.host.wall_sub_face_area
        glz_areas = []
        for seg, glz in zip(self.floor_segments, self.comparison_windows):
            if glz is not None:
                area = glz.area_from_segment(seg, self.host.floor_to_ceiling_height)
                glz_areas.append(area)
        return sum(glz_areas)

    @property
    def wall_sub_face_area_difference(self):
        """Get a number for the difference between host and comparison wall sub-face area.

        This number will be positive if the sub-face area increased in the host room
        compared to the comparison room and negative if it decreased.
        """
        return self.host.wall_sub_face_area - self.wall_sub_face_area

    @property
    def wall_sub_face_area_abs_difference(self):
        """Get a number for the difference between host and comparison wall sub-face area.
        """
        return abs(self.wall_sub_face_area_difference)

    @property
    def wall_sub_face_area_percent_change(self):
        """Get a number between 0 an 100 for the percent change between wall sub-face areas.
        """
        return (self.wall_sub_face_area_abs_difference / self.wall_sub_face_area) * 100

    @property
    def roof_sub_face_area(self):
        """Get a the total sub-face area of the comparison Room's roofs.

        This includes both Apertures and overhead Doors.
        """
        if self.host.is_top_exposed and self.comparison_skylight is not None and \
                self.comparison_floor_geometry is not None:
            sky_par = self.comparison_skylight
            return sky_par.area_from_face(self.comparison_floor_geometry)
        return self.host.roof_sub_face_area

    @property
    def roof_sub_face_area_difference(self):
        """Get a number for the difference between host and comparison roof sub-face area.

        This number will be positive if the sub-face area increased in the host room
        compared to the comparison room and negative if it decreased.
        """
        return self.host.roof_sub_face_area - self.roof_sub_face_area

    @property
    def roof_sub_face_area_abs_difference(self):
        """Get a number for the difference between host and comparison roof sub-face area.
        """
        return abs(self.roof_sub_face_area_difference)

    @property
    def roof_sub_face_area_percent_change(self):
        """Get a number between 0 an 100 for the percent change between roof sub-face areas.
        """
        return (self.roof_sub_face_area_abs_difference / self.roof_sub_face_area) * 100

    @property
    def sub_face_area(self):
        """Get a the total sub-face area of the comparison Room.

        This includes both Apertures and overhead Doors.
        """
        return self.wall_sub_face_area + self.roof_sub_face_area

    @property
    def sub_face_area_difference(self):
        """Get a number for the difference between host and comparison sub-face area.

        This number will be positive if the sub-face area increased in the host room
        compared to the comparison room and negative if it decreased.
        """
        return self.wall_sub_face_area_difference + self.roof_sub_face_area_difference

    @property
    def sub_face_area_abs_difference(self):
        """Get a number for the difference between host and comparison sub-face area.
        """
        return abs(self.sub_face_area_difference)

    @property
    def sub_face_area_percent_change(self):
        """Get a number between 0 an 100 for the percent change between sub-face areas.
        """
        return (self.sub_face_area_abs_difference / self.sub_face_area) * 100

    def set_from_room_2d(self, comparison_room_2d):
        """Set the attributes of this Room2DComparisonProperties using a Room2D.

        Args:
            comparison_room_2d: A Room2D to which the host Room2D is being compared.
        """
        self.comparison_floor_geometry = comparison_room_2d.floor_geometry
        self.comparison_windows = comparison_room_2d.window_parameters
        self.comparison_skylight = comparison_room_2d.skylight_parameters

    def reset(self):
        """Reset the comparison attributes using the host Room2D."""
        self.comparison_floor_geometry = self.host.floor_geometry
        self.comparison_windows = self.host.window_parameters
        self.comparison_skylight = self.host.skylight_parameters

    def restore(self):
        """Get a Room2D with host properties and geometry restored from the comparison.

        The restored Room2D returned from this method will have all boundary
        conditions reset to outdoors and all shading parameters removed. Otherwise,
        all properties of the returned Room2D will match the host and all geometry
        will match the comparison.
        """
        # grab the relevant properties from the host Room2D
        room_2d_class = self.host.__class__
        identifier = self.host.identifier
        floor_geo = self.comparison_floor_geometry \
            if self.comparison_floor_geometry is not None else self.host.floor_geometry
        ftc = self.host.floor_to_ceiling_height
        w_par = self.comparison_windows \
            if self.comparison_windows is not None else self.host.window_parameters
        ground = self.host.is_ground_contact
        exposed = self.host.is_top_exposed
        new_room = room_2d_class(identifier, floor_geo, ftc, window_parameters=w_par,
                                 is_ground_contact=ground, is_top_exposed=exposed)

        # assign any additional properties to the new room
        if self.comparison_skylight is not None:
            new_room._skylight_parameters = self.comparison_skylight
        new_room._has_floor = self.host._has_floor
        new_room._has_ceiling = self.host._has_ceiling
        new_room._ceiling_plenum_depth = self.host._ceiling_plenum_depth
        new_room._floor_plenum_depth = self.host._floor_plenum_depth
        new_room._user_data = None if self.host.user_data is None else \
            self.host.user_data.copy()
        new_room._parent = self.host._parent
        new_room._abridged_properties = self.host._abridged_properties
        new_room._properties._duplicate_extension_attr(self.host._properties)

        return new_room

    def move(self, moving_vec):
        """Move these properties along a vector.

        Args:
            moving_vec: A ladybug_geometry Vector3D with the direction and distance
                to move the room.
        """
        if self.comparison_floor_geometry is not None:
            self.comparison_floor_geometry = \
                self.comparison_floor_geometry.move(moving_vec)
        if isinstance(self.comparison_skylight, DetailedSkylights):
            self.comparison_skylight = self.comparison_skylight.move(moving_vec)

    def rotate_xy(self, angle, origin):
        """Rotate these properties counterclockwise in the XY plane by a certain angle.

        Args:
            angle: An angle in degrees.
            origin: A ladybug_geometry Point3D for the origin around which the
                object will be rotated.
        """
        if self.comparison_floor_geometry is not None:
            self.comparison_floor_geometry = \
                self.comparison_floor_geometry.rotate_xy(math.radians(angle), origin)
        if isinstance(self.comparison_skylight, DetailedSkylights):
            self.comparison_skylight = self.comparison_skylight.rotate(angle, origin)

    def scale(self, factor, origin=None):
        """Scale these properties by a factor from an origin point.

        Args:
            factor: A number representing how much the object should be scaled.
            origin: A ladybug_geometry Point3D representing the origin from which
                to scale. If None, it will be scaled from the World origin (0, 0, 0).
        """
        # scale the floor geometry
        if self.comparison_floor_geometry is not None:
            self.comparison_floor_geometry = \
                self.comparison_floor_geometry.scale(factor, origin)
        # scale the window parameters
        if self.comparison_windows is not None:
            scaled_windows = []
            for win_par in self.comparison_windows:
                s_wp = win_par.scale(factor) if win_par is not None else None
                scaled_windows.append(s_wp)
            self.comparison_windows = tuple(scaled_windows)
        # scale the skylight parameters
        if self.comparison_skylight is not None:
            self.comparison_skylight = self.comparison_skylight.scale(factor, origin) \
                if isinstance(self.comparison_skylight, DetailedSkylights) else \
                self.comparison_skylight.scale(factor)

    @classmethod
    def from_dict(cls, data, host):
        """Create Room2DComparisonProperties from a dictionary.

        Args:
            data: A dictionary representation of Room2DComparisonProperties in the
                format below.
            host: A Room2D object that hosts these properties.

        .. code-block:: python

            {
            "type": 'Room2DComparisonProperties',
            "modifier_set": {},  # A ModifierSet dictionary
            "grid_parameters": []  # A list of GridParameter dictionaries
            }
        """
        assert data['type'] == 'Room2DComparisonProperties', \
            'Expected Room2DComparisonProperties. Got {}.'.format(data['type'])
        new_prop = cls(host)
        new_prop.apply_properties_from_dict(data)
        return new_prop

    def apply_properties_from_dict(self, data):
        """Apply properties from a Room2DComparisonProperties dictionary.

        Args:
            data: A Room2DComparisonProperties dict (typically coming from a Model).
        """
        # re-assemble the floor_geometry
        if 'floor_boundary' in data and data['floor_boundary'] is not None:
            fh = self.host.floor_height
            bound_verts = [Point3D(pt[0], pt[1], fh) for pt in data['floor_boundary']]
            if 'floor_holes' in data:
                hole_verts = [[Point3D(pt[0], pt[1], fh) for pt in hole]
                              for hole in data['floor_holes']]
            else:
                hole_verts = None
            self.comparison_floor_geometry = Face3D(bound_verts, None, hole_verts)

        # re-assemble window parameters
        if 'window_parameters' in data and data['window_parameters'] is not None:
            glz_pars = []
            for glz_dict in data['window_parameters']:
                if glz_dict is not None:
                    try:
                        glz_class = getattr(glzpar, glz_dict['type'])
                    except AttributeError:
                        raise ValueError(
                            'Window parameter "{}" is not recognized.'.format(
                                glz_dict['type']))
                    glz_pars.append(glz_class.from_dict(glz_dict))
                else:
                    glz_pars.append(None)
            self.comparison_windows = glz_pars

        # assign any skylight parameters if they are specified
        if 'skylight_parameters' in data and data['skylight_parameters'] is not None:
            try:
                sky_class = getattr(skypar, data['skylight_parameters']['type'])
            except AttributeError:
                raise ValueError(
                    'Skylight parameter "{}" is not recognized.'.format(
                        data['skylight_parameters']['type']))
            self.comparison_skylight = sky_class.from_dict(data['skylight_parameters'])

    def to_dict(self, abridged=False):
        """Return Room2D comparison properties as a dictionary.

        Args:
            abridged: Boolean for whether the full dictionary of the Room2D should
                be written (False) or just the identifier of the the individual
                properties (True). Default: False.
        """
        base = {'comparison': {}}
        base['comparison']['type'] = 'Room2DComparisonProperties'

        # write the floor geometry into the dictionary
        if self.comparison_floor_geometry is not None:
            base['comparison']['floor_boundary'] = \
                [(p.x, p.y) for p in self.comparison_floor_geometry.boundary]
            if self.comparison_floor_geometry.has_holes:
                base['comparison']['floor_holes'] = \
                    [[(p.x, p.y) for p in hole]
                     for hole in self.comparison_floor_geometry.holes]

        # write the window parameters into the dictionary
        if self.comparison_windows is not None:
            base['comparison']['window_parameters'] = []
            for glz in self.comparison_windows:
                val = glz.to_dict() if glz is not None else None
                base['comparison']['window_parameters'].append(val)

        # write the skylights into the dict
        if self.comparison_skylight is not None:
            base['comparison']['skylight_parameters'] = \
                self.comparison_skylight.to_dict()
        return base

    def duplicate(self, new_host=None):
        """Get a copy of this object.

        Args:
            new_host: A new Room2D object that hosts these properties.
                If None, the properties will be duplicated with the same host.
        """
        _host = new_host or self._host
        new_r = Room2DComparisonProperties(_host, self._comparison_floor_geometry)
        new_r._comparison_windows = self._comparison_windows
        new_r._comparison_skylight = self._comparison_skylight
        return new_r

    def ToString(self):
        return self.__repr__()

    def __repr__(self):
        return 'Room2D Comparison Properties: {}'.format(self.host.identifier)
