"""Display attribute objects for creating visualization sets."""
from honeybee.facetype import Wall, RoofCeiling, Floor, AirBoundary
from honeybee.aperture import Aperture
from honeybee.shade import Shade
from honeybee.boundarycondition import Outdoors, Surface, Ground


class RoomAttribute(object):
    """A Room attribute object.

    Args:
        name: A name for this Room Attribute.
        attrs: A list of text strings of attributes that the Model Rooms have, which will
            be used to construct a visualization of this attribute in the resulting
            VisualizationSet. This can also be a list of attribute strings and a
            separate VisualizationData will be added to the AnalysisGeometry that
            represents the attribute in the resulting VisualizationSet (or a separate
            ContextGeometry layer if color is True). Attributes input here can have '.'
            that separates the nested attributes from one another. For example,
            'properties.energy.construction' or 'user_data.tag'
        color: A boolean to note whether the input room_attr should be expressed as a
            colored AnalysisGeometry. (Default: True)
        text: A boolean to note whether the input room_attr should be expressed as a
            a ContextGeometry as text labels. (Default: False)
        legend_par:An optional LegendParameter object to customize the display of the
            attribute. For text attribute only the text_height and font will be used to
            customize the text.
    """

    def __init__(
        self, name, attrs, color=True, text=False, legend_par=None
            ):
        self.name = name
        self.attrs = attrs
        self.color = color
        self.text = text
        self.legend_par = legend_par

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        self._attrs = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def legend_par(self):
        return self._legend_par

    @legend_par.setter
    def legend_par(self, value):
        self._legend_par = value


class FaceAttribute(RoomAttribute):
    """A Face attribute object.

    Args:
        name: A name for this Face Attribute.
        attrs: A list of text strings of attributes the Model Faces have, which will
            be used to construct a visualization of this attribute in the resulting
            VisualizationSet. This can also be a list of attribute strings and a
            separate VisualizationData will be added to the AnalysisGeometry that
            represents the attribute in the resulting VisualizationSet (or a separate
            ContextGeometry layer if color is True). Attributes input here can have '.'
            that separates the nested attributes from one another. For example,
            'properties.energy.construction' or 'user_data.tag'

        color: A boolean to note whether the input room_attr should be expressed as a
            colored AnalysisGeometry. (Default: True)

        text: A boolean to note whether the input room_attr should be expressed as a
            a ContextGeometry as text labels. (Default: False)

        legend_par:An optional LegendParameter object to customize the display of the
            attribute. For text attribute only the text_height and font will be used to
            customize the text.

        face_types: List of face types to be included in the visualization set. By
            default all the faces will be exported to visualization set. Valid values
            are:

            * Wall
            * RoofCeiling
            * Floor
            * AirBoundary
            * Aperture
            * Shade

        boundary_conditions: List of face boundary conditions to be included in the
            visualization set. This condition will be applied as a secondary check for
            the face_types that are set using the face_types argument. Valid values
            are:

            * Outdoors
            * Surface
            * Ground

    """

    def __init__(
        self, name, attrs, color=True, text=False, legend_par=None, face_types=None,
            boundary_conditions=None):
        super().__init__(name, attrs, color, text, legend_par)
        self.face_types = face_types
        self.boundary_conditions = boundary_conditions

    @property
    def face_types(self):
        return self._face_types

    @face_types.setter
    def face_types(self, types):
        types = types or []
        for type in types:
            assert type in (Wall, RoofCeiling, Floor, AirBoundary, Aperture, Shade), \
                f'Invalid face type: {type}'
        self._face_types = types

    @property
    def boundary_conditions(self):
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, bcs):
        bcs = bcs or []
        for bc in bcs:
            assert bc in (Outdoors, Surface, Ground), \
                f'Invalid face boundary condition: {bc}. Valid values are Outdoors, ' \
                'Surface and Ground.'
        self._boundary_conditions = bcs
