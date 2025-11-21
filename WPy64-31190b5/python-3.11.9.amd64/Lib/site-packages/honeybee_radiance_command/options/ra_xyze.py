# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    NumericOption,
    TupleOption
)


class Ra_xyzeOptions(OptionCollection):
    """ra_xyze command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/ra_xyze.1.html
    """

    __slots__ = (
        "_r",
        "_e",
        "_o",
        "_c",
        "_u",
        "_p"
        )

    def __init__(self):
        """ra_xyze command options."""
        OptionCollection.__init__(self)
        self._r = BoolOption("r", "produce run-length encoded RGBE")
        self._e = NumericOption("e", "exposure adjustment multiplier")
        self._o = BoolOption("o", "original units to which exposure is applied")
        self._c = BoolOption("c", "produce run-length encoded XYZE")
        self._u = BoolOption("u", "produce flat output")
        self._p = TupleOption(
            "p", "override standard Radiance RGB primary colors", None, 8, float
        )
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection. For
        instance in ra_xyze option collection -c and -u don't go together very well.
        You can include a check to ensure this is always correct.
        """

        if self._c.is_set and self._u.is_set:
            raise ValueError(
                'Both c and u are set. The program can use either c or u but not both.')

    @property
    def r(self):
        """Produce run-length encoded RGBE"""
        return self._r

    @r.setter
    def r(self, value):
        self._r.value = value

    @property
    def e(self):
        """Exposure adjustment multiplier

        Adjust the exposure. If exp is preceded by a '+' or '-', the exposure is
        interpreted in f-stops (ie. the power of two). Otherwise, exp is
        interpreted as a straight multiplier.
        """
        return self._e

    @e.setter
    def e(self, value):
        self._e.value = value

    @property
    def o(self):
        """Original units to which exposure compensation is applied"""
        return self._o

    @o.setter
    def o(self, value):
        self._o.value = value

    @property
    def c(self):
        """Produce run-length encoded XYZE"""
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value

    @property
    def u(self):
        """Produce flat output"""
        return self._u

    @u.setter
    def u(self, value):
        self._u.value = value

    @property
    def p(self):
        """Override standard Radiance RGB primary colors

        The eight floating-point arguments to this option are the 1931 CIE (x,y) 
        chromaticity coordinates of the three RGB primaries plus the white point, in that
        order. The new primaries will be recorded in the header of the output file, so 
        that the original information may be fully recovered later.
        """
        return self._p

    @p.setter
    def p(self, value):
        self._p.value = value
