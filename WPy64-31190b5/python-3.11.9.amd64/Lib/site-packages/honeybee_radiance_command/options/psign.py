# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    IntegerOption,
    NumericOption,
    FileOption,
    TupleOption
)


class PsignOptions(OptionCollection):
    """psign command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/psign.1.html
    """

    __slots__ = (
        "_cb",
        "_cf",
        "_dr",
        "_du",
        "_dl",
        "_dd",
        "_h",
        "_a",
        "_x",
        "_y",
        "_s",
        "_f"
        )

    def __init__(self):
        """psign command options."""

        OptionCollection.__init__(self)

        self._cb = TupleOption(
            "cb", "Background RGB - default: (1, 1, 1)", length=3, value=None,
            numtype=float)
        self._cf = TupleOption(
            "cf", "Foreground RGB - default: (0, 0, 0)", length=3, value=None,
            numtype=float)
        self._dr = BoolOption("dr", "Text reads to the right - default: True")
        self._du = BoolOption("du", "Text reads upwards - default: False")
        self._dl = BoolOption("dl", "Text reads left (upside down) - default: False")
        self._dd = BoolOption("dd", "Text reads downwards - default: False")
        self._h = IntegerOption("h", "Character height - default: 32 pixels")
        self._a = NumericOption("a", "Character aspect ratio (h/w) - default: 1.67")
        self._x = IntegerOption("x", "Horizontal image size in pixels")
        self._y = IntegerOption("y", "Vertical image size in pixels")
        self._s = NumericOption("s", "Character spacing - default: 0")
        self._f = FileOption("f", "Font file - default lib/helvet.fnt")

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection. For
        instance in pcond option collection -f and -p don't go together very well.
        You can include a check to ensure this is always correct.
        """
        all_orient = [1 for i in (self._dr, self._du, self._dl, self._dd) if i.is_set]
        if sum(all_orient) > 1:
            raise ValueError(
                'Only one of the -dr, -du, -dl, -dd options can be set at a time.'
            )

    @property
    def cb(self):
        """Background RGB color - default: (1, 1, 1) for white"""
        return self._cb

    @cb.setter
    def cb(self, value):
        self._cb.value = value

    @property
    def cf(self):
        """Foreground RGB color - default: (0, 0, 0) for black"""
        return self._cf

    @cf.setter
    def cf(self, value):
        self._cf.value = value

    @property
    def dr(self):
        """Text reads to the right - default: True"""
        return self._dr

    @dr.setter
    def dr(self, value):
        self._dr.value = value

    @property
    def du(self):
        """Text reads upwards - default: False"""
        return self._du

    @du.setter
    def du(self, value):
        self._du.value = value

    @property
    def dl(self):
        """Text reads to the left - default: False"""
        return self._dl

    @dl.setter
    def dl(self, value):
        self._dl.value = value

    @property
    def dd(self):
        """Text reads downwards - default: False"""
        return self._dd

    @dd.setter
    def dd(self, value):
        self._dd.value = value

    @property
    def h(self):
        """Character height - default: 32 pixels"""
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def a(self):
        """Character aspect ratio (height/width) - default: 1.67"""
        return self._a

    @a.setter
    def a(self, value):
        self._a.value = value

    @property
    def x(self):
        """Horizontal image size in pixels

        Use with −y option in place of the −h specification to control output
        image size directly. If the character aspect ratio (−a option, above)
        is non-zero, then one of the specified x or y output dimensions may be
        reduced to maintain this ratio. If direction is right (−dr) or left (−dl),
        then it is not necessary to give the −y option, since it can be computed
        from the character height (−h).
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value

    @property
    def y(self):
        """Vertical image size in pixels

        Use with the −x option. If direction is up (−du) or down (−dd), then it
        is not necessary to give the −x option, since it can be computed from
        the character height (−h).
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y.value = value

    @property
    def s(self):
        """Character spacing - default: 0

        The magnitude of this value is multiplied by the character height over the
        aspect ratio (ie. the character width) to compute the desired distance between
        characters in the output. The sign of the value, positive or negative,
        determines how this ideal spacing is used in the actual placement of
        characters. If spacing is positive, then the overall width of the line
        will not be affected, nor will indentation of textual elements. Thus, the
        text format will be mostly unaffected. However, spacing between characters
        will reflect their relative size for a more natural appearance. If spacing
        is negative, characters will be squeezed together to meet the spacing
        criterion, regardless of how it might affect the format of the output. The
        default value for spacing is zero, which is interpreted as uniformly
        spaced characters.
        """
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def f(self):
        """Font file - default lib/helvet.fnt"""
        return self._f

    @f.setter
    def f(self, value):
        self._f.value = value
