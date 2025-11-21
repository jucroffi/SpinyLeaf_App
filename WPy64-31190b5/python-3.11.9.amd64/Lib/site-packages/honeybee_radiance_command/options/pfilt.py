# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    StringOption,
    IntegerOption,
    NumericOption,
    FileOption
)


class PfiltOptions(OptionCollection):
    """pfilt command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/pfilt.1.html
    """

    __slots__ = (
        "_x",
        "_y",
        "_p",
        "_c",
        "_e",
        "_er",
        "_eg",
        "_eb",
        "_t",
        "_f",
        "_1_",
        "_2_",
        "_b",
        "_r",
        "_m",
        "_h",
        "_n",
        "_s",
        "_a"
        )

    def __init__(self):
        """pfilt command options."""

        OptionCollection.__init__(self)

        self._x = StringOption("x", "X resolution - default: same as input")
        self._y = StringOption("y", "Y resolution - default: same as input")
        self._p = NumericOption("p", "Pixel aspect ratio - default: 0")
        self._c = BoolOption("c", "Do not write PIXASPECT variable - default: False")
        self._e = NumericOption("e", "Exposure adjustment multiplier - default: 1")
        self._er = NumericOption("er", "Red exposure adjustment multiplier")
        self._eg = NumericOption("eg", "Green exposure adjustment multiplier")
        self._eb = NumericOption("eb", "Blue exposure adjustment multiplier")
        self._t = StringOption("t", "Fixture type to color balance")
        self._f = FileOption("f", "Lamp lookup table file - default lib/lamp.tab")
        self._1_ = BoolOption("_1", "Use only one pass on the file - default: False")
        self._2_ = BoolOption("_2", "Use two passes on the file - default: True")
        self._b = BoolOption("b", "Use box filtering - default: True")
        self._r = NumericOption("r", "Use Gaussian filtering with specified radius")
        self._m = NumericOption("m", "Limit given input pixels by a fraction")
        self._h = NumericOption("h", "Pixel intensity considered 'hot' - default: 100")
        self._n = IntegerOption("n", "Number of points on star patterns - default 0")
        self._s = NumericOption("s", "Spread for star patterns - default: .0001")
        self._a = BoolOption("a", "Average hot spots - default: False")

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection. For
        instance in pcond option collection -f and -p don't go together very well.
        You can include a check to ensure this is always correct.
        """

        if self._e.is_set and (self._er.is_set or self._eg.is_set or self._eb.is_set):
            raise ValueError(
                'Both -e and -er/-eg/-eb do not go well together.'
                ' This program can use either of the options but not both.'
            )

    @property
    def x(self):
        """X resolution - default: same as input

        Set the output x resolution to a number or to a division of the input resolution.
        Numbers must be less than or equal to the x dimension of the target device.
        If res is given as a slash followed by a real number (eg. /2), the input
        resolution is divided by this number to get the output resolution. By default,
        the output resolution is the same as the input.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value

    @property
    def y(self):
        """Y resolution - default: same as input

        Set the output y resolution to a number or to a division of the input resolution.
        Numbers must be less than or equal to the x dimension of the target device.
        If res is given as a slash followed by a real number (eg. /2), the input
        resolution is divided by this number to get the output resolution. By default,
        the output resolution is the same as the input.
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y.value = value

    @property
    def p(self):
        """Pixel aspect ratio - default: 0

        Either the x or the y resolution will be reduced so that the pixels have
        this ratio for the specified picture. If rat is zero, then the x and y
        resolutions will adhere to the given maxima. Zero is the default.
        """
        return self._p

    @p.setter
    def p(self, value):
        self._p.value = value

    @property
    def c(self):
        """Do not write PIXASPECT variable - default: False

        Pixel aspect ratio is being corrected, so do not write PIXASPECT variable
        to output file.
        """
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value

    @property
    def e(self):
        """Exposure adjustment multiplier - default: 1

        Adjust the exposure. If exp is preceded by a '+' or '-', the exposure is
        interpreted in f-stops (ie. the power of two). Otherwise, exp is
        interpreted as a straight multiplier.
        """
        return self._e

    @e.setter
    def e(self, value):
        self._e.value = value

    @property
    def er(self):
        """Exposure adjustment multiplier for Red channel."""
        return self._er

    @er.setter
    def er(self, value):
        self._er.value = value

    @property
    def eg(self):
        """Exposure adjustment multiplier for Green channel."""
        return self._eg

    @eg.setter
    def eg(self, value):
        self._eg.value = value

    @property
    def eb(self):
        """Exposure adjustment multiplier for Blue channel."""
        return self._eb

    @eb.setter
    def eb(self, value):
        self._eb.value = value

    @property
    def t(self):
        """Fixture type to color balance.

        Color-balance the image as if it were illuminated by fixtures of the
        given type. The specification must match a pattern listed in the lamp
        lookup table (see the −f option below).
        """
        return self._t

    @t.setter
    def t(self, value):
        self._t.value = value

    @property
    def f(self):
        """Lamp lookup table file - default lib.lamp.tab

        Use the specified lamp lookup table rather than the default (lamp.tab).
        """
        return self._f

    @f.setter
    def f(self, value):
        self._f.value = value

    @property
    def _1(self):
        """Use only one pass on the file - default: False

        This allows the exposure to be controlled absolutely, without any
        averaging. Note that a single pass is much quicker and should be
        used whenever the desired exposure is known and star patterns
        are not required.
        """
        return self._1_

    @_1.setter
    def _1(self, value):
        self._1_.value = value

    @property
    def _2(self):
        """Use two passes on the input - default: True"""
        return self._2_

    @_2.setter
    def _2(self, value):
        self._2_.value = value

    @property
    def b(self):
        """Use box filtering - default: True

        Box filtering averages the input pixels corresponding to each separate
        output pixel.
        """
        return self._b

    @b.setter
    def b(self, value):
        self._b.value = value

    @property
    def r(self):
        """Use Gaussian filtering with specified radius

        Use Gaussian filtering with a specified radius relative to the output
        pixel size. This option with a radius around 0.6 and a reduction in
        image width and height of 2 or 3 produces the highest quality
        pictures. A radius greater than 0.7 results in a defocused picture.
        """
        return self._r

    @r.setter
    def r(self, value):
        self._r.value = value

    @property
    def m(self):
        """Limit given input pixels by a fraction

        Limit the influence of any given input pixel to a fraction of any given output
        pixel. This option may be used to mitigate the problems associated with
        inadequate image sampling, at the expense of a slightly blurred image.
        The fraction given should not be less than the output picture dimensions
        over the input picture dimensions (x_o*y_o/x_i/y_i), or blurring will
        occur over the entire image. This option implies the −r option for Gaussian
        filtering, which defaults to a radius of 0.6.
        """
        return self._m

    @m.setter
    def m(self, value):
        self._m.value = value

    @property
    def h(self):
        """Pixel intensity considered "hot" - default: 100 watts/sr/m2

        Set intensity considered "hot" to a value. This is the level above which
        areas of the image will begin to exhibit star diffraction patterns (see below).
        The default is 100 watts/sr/m2.
        """
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def n(self):
        """Number of points on star patterns - default 0

        Set the number of points on star patterns to N. A value of zero turns
        star patterns off. The default is 0. (Note that two passes are required
        for star patterns.)
        """
        return self._n

    @n.setter
    def n(self, value):
        self._n.value = value

    @property
    def s(self):
        """Spread for star patterns - default: .0001

        Set the spread for star patterns to val. This is the value a star pattern will
        have at the edge of the image. The default is .0001.
        """
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def a(self):
        """Average hot spots - default: False

        By default, the areas of the picture above the hot level are not used in
        setting the exposure.
        """
        return self._a

    @a.setter
    def a(self, value):
        self._a.value = value
