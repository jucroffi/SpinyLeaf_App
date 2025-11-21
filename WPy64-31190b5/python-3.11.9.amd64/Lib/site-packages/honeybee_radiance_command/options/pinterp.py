# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    StringOption,
    IntegerOption,
    NumericOption,
)


class PinterpOptions(OptionCollection):
    """pinterp command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/pinterp.1.html
    """

    __slots__ = (
        "_vf",
        "_x",
        "_y",
        "_t",
        "_fa",
        "_ff",
        "_fb",
        "_f0",
        "_n",
        "_e",
        )

    def __init__(self):
        """pinterp command options."""
        OptionCollection.__init__(self)
        self._vf = StringOption("vf", "view options")
        self._x = IntegerOption("x", "x resolution")
        self._y = IntegerOption("y", "y resolution")
        self._t = NumericOption("t", "threshold for coincident pixels - default: 0.02")
        self._fa = BoolOption("fa", "foreground and background filling")
        self._ff = BoolOption("ff", "foreground filling")
        self._fb = BoolOption("fb", "background filling")
        self._f0 = BoolOption("f0", "no fill algorithm")
        self._n = BoolOption("n", "z distances along view direction")
        self._e = NumericOption("e", "exposure adjustment multiplier - default: 1")

    @property
    def vf(self):
        """View options

        Set the view to extract.
        """
        return self._vf

    @vf.setter
    def vf(self, value):
        self._vf.value = value

    @property
    def x(self):
        """x resolution

        Set the maximum x resolution.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value

    @property
    def y(self):
        """y resolution

        Set the maximum y resolution.
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y.value = value

    @property
    def t(self):
        """Threshold for coincident pixels - default: 0.02

        Pixels that map within the −t threshold of each other (.02 times the z distance 
        by default) are considered coincident.
        """
        return self._t

    @t.setter
    def t(self, value):
        self._t.value = value

    @property
    def fa(self):
        """Enable foreground and background filling"""
        return self._fa

    @fa.setter
    def fa(self, value):
        self.fa.value = value

    @property
    def ff(self):
        """Enable foreground filling"""
        return self._ff

    @ff.setter
    def ff(self, value):
        self.ff.value = value

    @property
    def fb(self):
        """Enable background filling"""
        return self._fb

    @fb.setter
    def fb(self, value):
        self.fb.value = value

    @property
    def f0(self):
        """Disable filling algorithm"""
        return self._f0

    @f0.setter
    def f0(self, value):
        self.f0.value = value

    @property
    def n(self):
        """specifies that input and output z distances are along the view direction, 
        rather than absolute distances to intersection points. This option is usually 
        appropriate with a constant z specification, and should not be used with rpict
        z files."""
        return self._n

    @n.setter
    def n(self, value):
        self.n.value = value

    @property
    def e(self):
        """Exposure adjustment multiplier - default: 1"""
        return self._e

    @e.setter
    def e(self, value):
        self._e.value = value
