# coding: utf-8
from .optionbase import (
    OptionCollection,
    BoolOption,
    NumericOption,
    IntegerOption,
    TupleOption,
)


class GendaylitOptions(OptionCollection):
    """Gendaylit command options.

    Also see: https://floyd.lbl.gov/radiance/gendaylit.1.html
    """

    __slots__ = (
        '_P',
        '_W',
        '_L',
        '_O',
        '_s',
        '_g',
        '_a',
        '_o',
        '_m',
        '_ang'
        )

    def __init__(self):
        """Gendaylit command options."""

        OptionCollection.__init__(self)

        self._P = TupleOption("P", "Espilon data a.k.a Perez parameters", length=2)
        self._W = TupleOption("W", "Direct normal irradiance and diffuse"
                              " horizontal irradiance", length=2, numtype=int)
        self._L = TupleOption("L", "Direct normal illuminance and"
                              " diffuse horizontal illuminance", length=2, numtype=int)
        self._O = IntegerOption("O", "Output", min_value=0, max_value=2)
        self._s = BoolOption("s", "Generate source description of the sun")
        self._g = NumericOption("g", "Average ground reflectance")
        self._a = NumericOption("a", "Site latitude", min_value=-90.0, max_value=90.0)
        self._o = NumericOption("o", "Site longitude", min_value=-180.0, max_value=180.0)
        self._m = NumericOption("m", "Standard meridian", min_value=-18.0,
                                max_value=18.0)
        self._ang = TupleOption("ang", "Altitude & azimuth", length=2, numtype=float)
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.
        Use this method to add checks that are necessary for OptionCollection.
        """
        if self._ang.is_set and (
                self._a.is_set or self._o.is_set or self._m.is_set):
            raise ValueError(
                'Options -a, -o and -m do not apply when -ang is set.'
            )

        params = {'perez': self._P.is_set, 'irradiance_values': self._W.is_set,
                  'illuminance_values': self._L.is_set}
        params_requested = [param for param in params if params[param] is True]

        if len(params_requested) > 1:
            raise ValueError(
                'Multiple parameters requested. Only either of %s parameters are'
                ' to be applied.' % (params_requested)
            )

    @property
    def P(self):
        """Espilon data a.k.a Perez parameters"""
        return self._P

    @P.setter
    def P(self, value):
        self._P.value = value

    @property
    def W(self):
        """A tuple of direct-normal-irradiance (W/m^2) and
        diffuse-horizontal-irradiance (W/m^2) values.
        """
        return self._W

    @W.setter
    def W(self, value):
        self._W.value = value

    @property
    def L(self):
        """A tuple of direct-normal-illuminance (lm/m^2) and
        diffuse-horizontal-illuminance (lm/m^2) values.
        """
        return self._L

    @L.setter
    def L(self, value):
        self._L.value = value

    @property
    def O(self):
        """Output

        A value of 0 will output in W/m^2/sr visible radiation,
        A value of 1 will output in W/m^2/sr solar radiation,
        A value of 2 will output in lm/m^2/sr luminance.
        """
        return self._O

    @O.setter
    def O(self, value):
        self._O.value = value

    @property
    def s(self):
        """Generate source description of the sun."""
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def g(self):
        """Average ground reflectance.

        This value is used to compute skyfunc when Dz is negative.
        """
        return self._g

    @g.setter
    def g(self, value):
        self._g.value = value

    @property
    def a(self):
        """Site latitude.

        The value is site latitude in degrees north.
        (Use negative angle for south latitude.)
        This is used in the calculation of sun angle.
        """
        return self._a

    @a.setter
    def a(self, value):
        self._a.value = value

    @property
    def o(self):
        """Site longitude.

        The value is site longitude in degrees west.
        (Use negative angle for east longitude.)
        This is used in the calculation of solar time and sun angle. Be sure to give
        the corresponding standard meridian also! If solar time is given directly,
        then this option has no effect.
        """
        return self._o

    @o.setter
    def o(self, value):
        self._o.value = value

    @property
    def m(self):
        """Standard meridian.

        The site standard meridian is a value in degrees west of Greenwich.
        (Use negative angle for east.) This is used in the calculation of solar time.
        Be sure to give the correct longitude also! If a time zone or solar time is
        given directly, then this option has no effect.
        """
        return self._m

    @m.setter
    def m(self, value):
        self._m.value = value

    @property
    def ang(self):
        """Altitude & azimuth

        This option gives the solar angles explicitly. The altitude is measured in
        degrees above the horizon, and the azimuth is measured in degrees
        west of south.
        """
        return self._ang

    @ang.setter
    def ang(self, value):
        self._ang.value = value
        
