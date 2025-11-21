# coding=utf-8
"""Room DOE-2 Properties."""
import math

from ladybug_geometry.geometry3d import Face3D
from honeybee.typing import float_in_range, float_positive
from honeybee.altnumber import autocalculate

from ..load import MECH_AIRFLOW_KEYS


class RoomDoe2Properties(object):
    """DOE-2 Properties for Honeybee Room.

    Args:
        host: A honeybee_core Room object that hosts these properties.
        assigned_flow: A number for the design supply air flow rate for the zone
            the Room is assigned to (cfm). This establishes the minimum allowed
            design air flow. Note that the actual design flow may be larger. If
            None, this parameter will not be written into the INP. (Default: None).
        flow_per_area: A number for the design supply air flow rate to
            the zone per unit floor area (cfm/ft2). If None, this parameter
            will not be written into the INP. (Default: None).
        min_flow_ratio: A number between 0 and 1 for the minimum allowable zone
            air supply flow rate, expressed as a fraction of design flow rate.
            Applicable to variable-volume type systems only. If None, this parameter
            will not be written into the INP. (Default: None).
        min_flow_per_area: A number for the minimum air flow per square foot of
            floor area (cfm/ft2). This is an alternative way of specifying the
            min_flow_ratio. If None, this parameter will not be written into
            the INP. (Default: None).
        hmax_flow_ratio: A number between 0 and 1 for the ratio of the maximum
            (or fixed) heating airflow to the cooling airflow. The specific
            meaning varies according to the type of zone terminal. If None, this
            parameter will not be written into the INP. (Default: None).
        space_polygon_geometry: An optional horizontal Face3D object, which will
            be used to set the SPACE polygon during export to INP. If None,
            the SPACE polygon is auto-calculated from the 3D Room geometry.
            Specifying a geometry here can help overcome some limitations of
            this auto-calculation, particularly for cases where the floors
            of the Room are composed of AirBoundaries. (Default: None).

    Properties:
        * host
        * assigned_flow
        * flow_per_area
        * min_flow_ratio
        * min_flow_per_area
        * hmax_flow_ratio
        * space_polygon_geometry
    """
    __slots__ = (
        '_host', '_assigned_flow', '_flow_per_area', '_min_flow_ratio',
        '_min_flow_per_area', '_hmax_flow_ratio', '_space_polygon_geometry'
    )

    def __init__(
        self, host, assigned_flow=None, flow_per_area=None, min_flow_ratio=None,
        min_flow_per_area=None, hmax_flow_ratio=None, space_polygon_geometry=None
    ):
        """Initialize Room DOE-2 properties."""
        # set the main properties of the Room
        self._host = host
        self.assigned_flow = assigned_flow
        self.flow_per_area = flow_per_area
        self.min_flow_ratio = min_flow_ratio
        self.min_flow_per_area = min_flow_per_area
        self.hmax_flow_ratio = hmax_flow_ratio
        self.space_polygon_geometry = space_polygon_geometry

    @property
    def host(self):
        """Get the Room object hosting these properties."""
        return self._host

    @property
    def assigned_flow(self):
        """Get or set the design supply air flow rate for the zone (cfm)."""
        return self._assigned_flow

    @assigned_flow.setter
    def assigned_flow(self, value):
        if value is not None:
            value = float_positive(value, 'zone assigned flow')
        self._assigned_flow = value

    @property
    def flow_per_area(self):
        """Get or set the design supply air flow rate per unit floor area (cfm/ft2).
        """
        return self._flow_per_area

    @flow_per_area.setter
    def flow_per_area(self, value):
        if value is not None:
            value = float_positive(value, 'zone flow per area')
        self._flow_per_area = value

    @property
    def min_flow_ratio(self):
        """Get or set the the min supply airflow rate as a fraction of design flow rate.
        """
        return self._min_flow_ratio

    @min_flow_ratio.setter
    def min_flow_ratio(self, value):
        if value is not None:
            value = float_in_range(value, 0.0, 1.0, 'zone min flow ratio')
        self._min_flow_ratio = value

    @property
    def min_flow_per_area(self):
        """Get or set the minimum air flow per square foot of floor area (cfm/ft2)."""
        return self._min_flow_per_area

    @min_flow_per_area.setter
    def min_flow_per_area(self, value):
        if value is not None:
            value = float_positive(value, 'zone min flow per area')
        self._min_flow_per_area = value

    @property
    def hmax_flow_ratio(self):
        """Get or set the ratio of the maximum heating airflow to the cooling airflow.
        """
        return self._hmax_flow_ratio

    @hmax_flow_ratio.setter
    def hmax_flow_ratio(self, value):
        if value is not None:
            value = float_in_range(value, 0.0, 1.0, 'zone heating max flow ratio')
        self._hmax_flow_ratio = value

    @property
    def space_polygon_geometry(self):
        """Get or set a horizontal Face3D to set the space polygon geometry."""
        return self._space_polygon_geometry

    @space_polygon_geometry.setter
    def space_polygon_geometry(self, value):
        if value is not None:
            assert isinstance(value, Face3D), \
                'Expected ladybug_geometry Face3D. Got {}'.format(type(value))
            if value.normal.z < 0:  # ensure upward-facing Face3D
                self._floor_geometry = value.flip()
        self._space_polygon_geometry = value

    def move(self, moving_vec):
        """Move this object along a vector.

        Args:
            moving_vec: A ladybug_geometry Vector3D with the direction and distance
                to move the object.
        """
        if self.space_polygon_geometry is not None:
            self._space_polygon_geometry = self.space_polygon_geometry.move(moving_vec)

    def rotate(self, angle, axis, origin):
        """Rotate this object by a certain angle around an axis and origin.

        Args:
            angle: An angle for rotation in degrees.
            axis: Rotation axis as a Vector3D.
            origin: A ladybug_geometry Point3D for the origin around which the
                object will be rotated.
        """
        if self.space_polygon_geometry is not None:
            self._space_polygon_geometry = \
                self.space_polygon_geometry.rotate(math.radians(angle), axis, origin)

    def rotate_xy(self, angle, origin):
        """Rotate this object counterclockwise in the world XY plane by a certain angle.

        Args:
            angle: An angle in degrees.
            origin: A ladybug_geometry Point3D for the origin around which the
                object will be rotated.
        """
        if self.space_polygon_geometry is not None:
            self._space_polygon_geometry = \
                self.space_polygon_geometry.rotate_xy(math.radians(angle), origin)

    def reflect(self, plane):
        """Reflect this object across a plane.

        Args:
            plane: A ladybug_geometry Plane across which the object will
                be reflected.
        """
        if self.space_polygon_geometry is not None:
            self._space_polygon_geometry = self.space_polygon_geometry.reflect(plane)

    def scale(self, factor, origin=None):
        """Scale this object by a factor from an origin point.

        Args:
            factor: A number representing how much the object should be scaled.
            origin: A ladybug_geometry Point3D representing the origin from which
                to scale. If None, it will be scaled from the World origin (0, 0, 0).
        """
        if self.space_polygon_geometry is not None:
            self._space_polygon_geometry = \
                self.space_polygon_geometry.scale(factor, origin)

    @classmethod
    def from_dict(cls, data, host):
        """Create RoomDoe2Properties from a dictionary.

        Args:
            data: A dictionary representation of RoomDoe2Properties with the
                format below.
            host: A Room object that hosts these properties.

        .. code-block:: python

            {
            "type": 'RoomDoe2Properties',
            "assigned_flow": 100,  # number in cfm
            "flow_per_area": 1,  # number in cfm/ft2
            "min_flow_ratio": 0.3, # number between 0 and 1
            "min_flow_per_area": 0.3, # number in cfm/ft2
            "hmax_flow_ratio": 0.3,  # number between 0 and 1
            "space_polygon_geometry": {}  # optional Face3D dictionary
            }
        """
        assert data['type'] == 'RoomDoe2Properties', \
            'Expected RoomDoe2Properties. Got {}.'.format(data['type'])
        new_prop = cls(host)
        auto_dict = autocalculate.to_dict()
        if 'assigned_flow' in data and data['assigned_flow'] != auto_dict:
            new_prop.assigned_flow = data['assigned_flow']
        if 'flow_per_area' in data and data['flow_per_area'] != auto_dict:
            new_prop.flow_per_area = data['flow_per_area']
        if 'min_flow_ratio' in data and data['min_flow_ratio'] != auto_dict:
            new_prop.min_flow_ratio = data['min_flow_ratio']
        if 'min_flow_per_area' in data and data['min_flow_per_area'] != auto_dict:
            new_prop.min_flow_per_area = data['min_flow_per_area']
        if 'hmax_flow_ratio' in data and data['hmax_flow_ratio'] != auto_dict:
            new_prop.hmax_flow_ratio = data['hmax_flow_ratio']
        if 'space_polygon_geometry' in data and \
                data['space_polygon_geometry'] is not None:
            new_prop.space_polygon_geometry = \
                Face3D.from_dict(data['space_polygon_geometry'])
        return new_prop

    def apply_properties_from_dict(self, data):
        """Apply properties from a RoomDoe2Properties dictionary.

        Args:
            data: A RoomDoe2Properties dictionary (typically coming from a Model).
        """
        auto_dict = autocalculate.to_dict()
        if 'assigned_flow' in data and data['assigned_flow'] != auto_dict:
            self.assigned_flow = data['assigned_flow']
        if 'flow_per_area' in data and data['flow_per_area'] != auto_dict:
            self.flow_per_area = data['flow_per_area']
        if 'min_flow_ratio' in data and data['min_flow_ratio'] != auto_dict:
            self.min_flow_ratio = data['min_flow_ratio']
        if 'min_flow_per_area' in data and data['min_flow_per_area'] != auto_dict:
            self.min_flow_per_area = data['min_flow_per_area']
        if 'hmax_flow_ratio' in data and data['hmax_flow_ratio'] != auto_dict:
            self.hmax_flow_ratio = data['hmax_flow_ratio']
        if 'space_polygon_geometry' in data and \
                data['space_polygon_geometry'] is not None:
            self.space_polygon_geometry = \
                Face3D.from_dict(data['space_polygon_geometry'])

    def apply_properties_from_user_data(self):
        """Apply properties from a the user_data assigned to the host room.

        For this method to successfully assign properties from user_data, the
        properties on this object must currently be None and the keys in
        user_data must use the INP convention for each of the attributes,
        which must be CAPITALIZED like the following:

        .. code-block:: python

            {
            "ASSIGNED-FLOW": 100,  # number in cfm
            "FLOW/AREA": 1,  # number in cfm/ft2
            "MIN-FLOW-RATIO": 0.3, # number between 0 and 1
            "MIN-FLOW/AREA": 0.3, # number in cfm/ft2
            "HMAX-FLOW-RATIO": 0.3  # number between 0 and 1
            }
        """
        attrs = ('assigned_flow', 'flow_per_area', 'min_flow_ratio',
                 'min_flow_per_area', 'hmax_flow_ratio')
        data = self.host.user_data
        if data is not None:
            for key, attr in zip(MECH_AIRFLOW_KEYS, attrs):
                if key in data and getattr(self, attr) is None:
                    try:
                        setattr(self, attr, data[key])
                    except Exception:
                        pass  # it's user_data; users are allowed to make mistakes

    def to_dict(self, abridged=False):
        """Return Room Doe2 properties as a dictionary."""
        base = {'doe2': {}}
        base['doe2']['type'] = 'RoomDoe2Properties'
        if self.assigned_flow is not None:
            base['doe2']['assigned_flow'] = self.assigned_flow
        if self.flow_per_area is not None:
            base['doe2']['flow_per_area'] = self.flow_per_area
        if self.min_flow_ratio is not None:
            base['doe2']['min_flow_ratio'] = self.min_flow_ratio
        if self.min_flow_per_area is not None:
            base['doe2']['min_flow_per_area'] = self.min_flow_per_area
        if self.hmax_flow_ratio is not None:
            base['doe2']['hmax_flow_ratio'] = self.hmax_flow_ratio
        if self.space_polygon_geometry is not None:
            base['doe2']['space_polygon_geometry'] = \
                self.space_polygon_geometry.to_dict()
        return base

    def to_inp(self):
        """Get RoomDoe2Properties as INP (Keywords, Values).

        Returns:
            A tuple with two elements.

            -   keywords: A list of text strings for keywords to assign to the room.

            -   values: A list of text strings that aligns with the keywords and
                denotes the value for each keyword.
        """
        keywords = []
        values = []
        attrs = ('assigned_flow', 'flow_per_area', 'min_flow_ratio',
                 'min_flow_per_area', 'hmax_flow_ratio')
        for key, attr in zip(MECH_AIRFLOW_KEYS, attrs):
            attr_value = getattr(self, attr)
            if attr_value is not None:
                keywords.append(key)
                values.append(attr_value)
        return keywords, values

    def duplicate(self, new_host=None):
        """Get a copy of this object.

        Args:
            new_host: A new Room object that hosts these properties.
                If None, the properties will be duplicated with the same host.
        """
        _host = new_host or self._host
        new_room = RoomDoe2Properties(
            _host, self.assigned_flow, self.flow_per_area, self.min_flow_ratio,
            self.min_flow_per_area, self.hmax_flow_ratio, self.space_polygon_geometry)
        return new_room

    def ToString(self):
        return self.__repr__()

    def __repr__(self):
        return 'Room DOE2 Properties: [host: {}]'.format(self.host.display_name)
