# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    NumericOption,
    TupleOption,
    ToggleOption
)
import warnings


class GenskyOptions(OptionCollection):
    """Gensky command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/gensky.1.html
    """

    __slots__ = (
        "_s",
        "_c",
        "_i",
        "_u",
        "_g",
        "_b",
        "_B",
        "_r",
        "_R",
        "_t",
        "_a",
        "_o",
        "_m",
        "_ang")

    def __init__(self):
        """Gensky command options."""

        OptionCollection.__init__(self)

        self._s = ToggleOption("s", "Sunny sky", value=None)
        self._c = BoolOption("c", "Cloudy sky")
        self._i = ToggleOption("i", "Intermediate sky", value=None)
        self._u = BoolOption("u", "Uniform cloudy sky")
        self._g = NumericOption("g", "Average ground reflectance")
        self._b = NumericOption("b", "Zenith brightness computed from sun and"
                                " sky turbidity.")
        self._B = NumericOption("B", "Zenith brightness computed from"
                                " horizontal diffuse irradiance.")
        self._r = NumericOption("r", "Solar radiance computed from solar altitude.")
        self._R = NumericOption("R", "Solar radiance computed from"
                                " horizontal direct irradiance.")
        self._t = NumericOption("t", "Turbuity factor", min_value=1.0)
        self._a = NumericOption("a", "Site latitude", min_value=-90.0, max_value=90.0)
        self._o = NumericOption("o", "Site longitude", min_value=-180.0,
                                max_value=180.0)
        self._m = NumericOption("m", "Standard meridian", min_value=-18.0,
                                max_value=18.0)
        self._ang = TupleOption("ang", "Altitude & azimuth", length=2,
                                numtype=float)
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection.
        """
        if self._ang.is_set and (
                self._a.is_set or self._o.is_set or self._m.is_set):
            warnings.warn(
                'Options -a, -o and -m do not apply when -ang is set.'
            )

        skies = {'sunny sky': self._s.is_set, 'cloudy sky': self._c.is_set,
                 'Intermediate sky': self._i.is_set, 'Uniform sky': self._u.is_set}
        skies_requested = [sky for sky in skies if skies[sky] is True]

        if len(skies_requested) > 1:
            raise ValueError(
                'Multiple skies requested. %s.'
                ' Only one allowed.' % (skies_requested)
            )

    @property
    def s(self):
        """Sunny sky.

        Setting it to '+' will create a sunny sky with sun. Setting it to '-' will
        create a sunny sky without sun. The sky distribution will correspond to a
        standard CIE clear day. If set to true, in addition to the sky distribution
        function, a source description of the sun is generated.
        """
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def c(self):
        """Cloudy sky.

        The sky distribution will correspond to a standard CIE overcast day.
        """
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value

    @property
    def i(self):
        """Intermediate sky.

        Setting it to '+' will create an intermediate sky with sun. Setting it to '-'
        will create an intermediate sky without sun. The sky will correspond to a
        standard CIE intermediate day. If set to true, in addition to the sky
        distribution, a (somewhat subdued) sun is generated.
        """
        return self._i

    @i.setter
    def i(self, value):
        self._i.value = value

    @property
    def u(self):
        """Uniform cloudy sky.

        The sky distribution will be completely uniform
        """
        return self._u

    @u.setter
    def u(self, value):
        self._u.value = value

    @property
    def g(self):
        """Average ground reflectance.

        This value is used to compute skyfunc when Dz is negative. Ground plane
        brightness is the same for −s as for +s. (Likewise for −i and +i)
        """
        return self._g

    @g.setter
    def g(self, value):
        self._g.value = value

    @property
    def b(self):
        """Zenith brightness computed from sun and sky turbidity.

        Zenith radiance (in watts/steradian/meter2) is normally computed from the sun
        angle and sky turbidity (for sunny sky). It can be given directly instead,
        using this option.
        """
        return self._b

    @b.setter
    def b(self, value):
        self._b.value = value

    @property
    def B(self):
        """Zenith brightness computed from horizontal diffuse irradiance.

        In this option, zenith brightness is computed from the horizontal diffuse
        irradiance (in watts/meter2).
        """
        return self._B

    @B.setter
    def B(self, value):
        self._B.value = value

    @property
    def r(self):
        """Solar radiance computed from solar altitude.

        The value is solar radiance. Solar radiance (in watts/steradian/meter2)
        is normally computed from the solar altitude. This option may be used to
        override the default calculation. If a value of zero is given, no sun
        description is produced, and the contribution of direct solar to ground
        brightness is neglected.
        """
        return self._r

    @r.setter
    def r(self, value):
        self._r.value = value

    @property
    def R(self):
        """Solar radiance computed from horizontal direct irradiance.

        Solar radiance is computed from the horizontal direct irradiance
        (in watts/meter2).
        """
        return self._R

    @R.setter
    def R(self, value):
        self._R.value = value

    @property
    def t(self):
        """Turbuity factor.

        The value is turbidity factor. Greater turbidity factors correspond to greater
        atmospheric scattering. A turbidity factor of 1.0 indicates an ideal clear
        atmosphere (i.e. a completely dark sky). Values less than 1.0 are physically
        impossible.
        """
        return self._t

    @t.setter
    def t(self, value):
        self._t.value = value

    @property
    def a(self):
        """Site latitude.

        The value is site latitude in degrees north.
        (Use negative angle for south latitude.)
        This is used in the calculation of sun angle
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
        degrees above the horizon, and the azimuth is measured in degrees west of south.
        """
        return self._ang

    @ang.setter
    def ang(self, value):
        self._ang.value = value
