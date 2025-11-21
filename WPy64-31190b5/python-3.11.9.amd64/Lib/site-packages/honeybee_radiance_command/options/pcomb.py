# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    NumericOption,
    StringOption,
    IntegerOption,
    TupleOption,
    FileOption
)


class PcombOptions(OptionCollection):
    """pcomb command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/pcomb.1.html
    """

    __slots__ = (
        "_h",
        "_w",
        "_x",
        "_y",
        "_f",
        "_e",
        "_o",
        "_s",
        "_c"
        )

    def __init__(self):
        """pcomd command options."""

        OptionCollection.__init__(self)

        self._h = BoolOption("h", "Reduce information header - default: False")
        self._w = BoolOption("w", "Suppress warning messages - default: False")
        self._x = IntegerOption("x", "X resolution", min_value=1)
        self._y = IntegerOption("y", "Y resolution", min_value=1)
        self._f = FileOption("f", "function file")
        self._e = StringOption("e", "Expression")
        self._o = BoolOption("o", "Use original pixel values - default: False")
        self._s = NumericOption("s", "Factor for linear combination")
        self._c = TupleOption("c", "RGB values", length=3, value=None, numtype=float)

    @property
    def h(self):
        """Reduce information header - default: False

        The −h option may be used to reduce the information header size, which can
        grow disproportionately after multiple runs of pcomb.
        """
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def w(self):
        """Suppress warning messages - default: False

        The −w option can be used to suppress warning messages about invalid
        calculations.
        """
        return self._w

    @w.setter
    def w(self, value):
        self._w.value = value

    @property
    def x(self):
        """X resolution

        The −x option can be used to specify the desired output resolution, xres, and
        can be expressions involving other constants such as xmax.
        The constants xres may also be specified in a file or expression.
        The default output resolution is the same as the input resolution.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value

    @property
    def y(self):
        """Y resolution

        The −y option can be used to specify the desired output resolution, yres, and
        can be expressions involving other constants such as ymax.
        The constants yres may also be specified in a file or expression.
        The default output resolution is the same as the input resolution.
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y.value = value

    @property
    def f(self):
        """function file

        This function file can be used to assign arbitrary mapping of coefficients.
        """
        return self._f

    @f.setter
    def f(self, value):
        self._f.value = value

    @property
    def e(self):
        """Expression

        This expression can be used to assign arbitrary mapping of coefficients.
        """
        return self._e

    @e.setter
    def e(self, value):
        self._e.value = value

    @property
    def o(self):
        """Use original pixel values

        The −o option indicates that original pixel values are to be used for the next
        picture, undoing any previous exposure changes or color correction.
        """
        return self._o

    @o.setter
    def o(self, value):
        self._o.value = value

    @property
    def s(self):
        """Factor for linear combination

        A factor used in the linear combination of pictures.
        """
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def c(self):
        """RGB values

        RGB values to be applied in the linear combination of pictures.
        """
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value
        
