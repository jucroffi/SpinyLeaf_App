# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    StringOption,
    IntegerOption,
    TupleOption
)


class PcomposOptions(OptionCollection):
    """pcompos command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/pcompos.1.html
    """

    __slots__ = (
        "_a",
        "_s",
        "_o",
        "_h",
        "_x",
        "_y",
        "_b",
        "_l",
        "_lh",
        "_la"
        )

    def __init__(self):
        """pcompos command options."""

        OptionCollection.__init__(self)

        self._a = IntegerOption("a", "Number of columns", min_value=1)
        self._s = IntegerOption("s", "Spacing between images")
        self._o = TupleOption(
            "o", "Non-zero anchor point for bottom left as (x0, y0)",
            length=2, value=None, numtype=int
        )
        self._h = BoolOption("h", "Reduce information header - default: False")
        self._x = IntegerOption("x", "X resolution - default: 0")
        self._y = IntegerOption("y", "Y resolution - default: 0")
        self._b = TupleOption("b", "Background RGB", length=3, value=None, numtype=float)
        self._l = StringOption("l", "Label for image")
        self._lh = IntegerOption("lh", "Label height - default: 24 pixels")
        self._la = BoolOption("la", "Label with file name - default: False")
        self._on_setattr_check = False

    @property
    def a(self):
        """Number of columns for which anchor points will be automatically computed.

        It is assumed that anchor points place successive pictures next to each other
        in ncols columns. The ordering will place the first picture in the lower
        left corner, the next just to the right of it, and so on for ncols pictures.
        Then, the next row up repeats the pattern until all the input pictures
        have been added to the output. If the pictures are of different size,
        pcompos will end up leaving some background areas in the output picture.
        There will also be an unfinished row at the top if the number of pictures
        is not evenly divided by ncols.
        """
        return self._a

    @a.setter
    def a(self, value):
        self._a.value = value

    @property
    def s(self):
        """Integer for pixel spacing between images.

        The −s option causes each image to be separated by at least N pixels.
        """
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def o(self):
        """Specify a nonzero anchor point for the bottom left image as (x0, y0).

        This should be specified as a tuple of 2 integer values, denoting the X
        and Y pixels of the origin.
        """
        return self._o

    @o.setter
    def o(self, value):
        self._o.value = value

    @property
    def h(self):
        """Reduce information header - default: False

        The −h option may be used to reduce the information header size, which can
        grow disproportionately after multiple runs of pcompos and/or pcomb.
        """
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def x(self):
        """X dimension around the output image - default: 0

        By default, the size of the output picture will be just large enough to
        encompass all the input files. By specifying a smaller dimension using
        the −x and −y options, input files can be cropped at the upper boundary.
        Specifying a larger dimension produces a border.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value

    @property
    def y(self):
        """Y dimension around the output image - default: 0

        By default, the size of the output picture will be just large enough to
        encompass all the input files. By specifying a smaller dimension using
        the −x and −y options, input files can be cropped at the upper boundary.
        Specifying a larger dimension produces a border.
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y.value = value

    @property
    def b(self):
        """RGB value for background color - default: (0, 0, 0)

        The background color appears wherever input files do not cover.
        """
        return self._b

    @b.setter
    def b(self, value):
        self._b.value = value

    @property
    def l(self):
        """Text for a label for the output picture."""
        return self._l

    @l.setter
    def l(self, value):
        self._l.value = value

    @property
    def lh(self):
        """Label height in pixels - default: 24"""
        return self._lh

    @lh.setter
    def lh(self, value):
        self._lh.value = value

    @property
    def la(self):
        """Label images automatically according by name.

        This is particularly useful in conjunction with the −a option for producing
        a catalog of images.
        """
        return self._la

    @la.setter
    def la(self, value):
        self._la.value = value
