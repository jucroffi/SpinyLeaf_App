# coding: utf-8

from .optionbase import OptionCollection, BoolOption, StringOption


class GetinfoOptions(OptionCollection):
    """getinfo options.

    Also see: https://floyd.lbl.gov/radiance/man_html/getinfo.1.html
    """

    __slots__ = ('_d', '_a')

    def __init__(self):
        """getinfo command options."""

        OptionCollection.__init__(self)
        self._d = BoolOption("d", "Print the dimensions instead - default: False")
        self._a = StringOption("a", "Text to add to the file header", pattern_out='"%s"')
        self._on_setattr_check = False

    @property
    def d(self):
        """Print the dimensions instead - default: False

        The −d option can be used to print the dimensions of an octree or picture
        file instead of getting the header. For an octree, getinfo −d prints the
        bounding cube (xmin ymin zmin size). For a picture, getinfo −d prints the
        y and x resolution (−Y yres +X xres).
        """
        return self._d

    @d.setter
    def d(self, value):
        self._d.value = value

    @property
    def a(self):
        """Text to add to the file header

        Add one or more header lines to the standard input. These lines are given
        as arguments to getinfo, and will be automatically quoted if they
        contain spaces.
        """
        return self._a

    @a.setter
    def a(self, value):
        self._a.value = value
