# coding: utf-8

from .optionbase import OptionCollection, BoolOption


class PflipOptions(OptionCollection):
    """pflip options.

    Also see: https://floyd.lbl.gov/radiance/man_html/pflip.1.html
    """

    __slots__ = ('_h', '_v', '_c')

    def __init__(self):
        """pflip command options."""

        OptionCollection.__init__(self)
        self._h = BoolOption("h", "Flip horizontally - default: False")
        self._v = BoolOption("v", "Clip vertically - default: False")
        self._c = BoolOption("c", "Correct improper image orientation - default: False")
        self._on_setattr_check = False

    @property
    def h(self):
        """Boolean for whether to perform a horizontal flip."""
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def v(self):
        """Boolean for whether to perform a vertical flip."""
        return self._v

    @v.setter
    def v(self, value):
        self._v.value = value

    @property
    def c(self):
        """Boolean for whether to correct an improper original image orientation.

        If selected, the recorded scanline ordering will not be changed.
        """
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value
