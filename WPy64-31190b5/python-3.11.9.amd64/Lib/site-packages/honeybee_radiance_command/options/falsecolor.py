# coding: utf-8

from .optionbase import (
    OptionCollection,
    BoolOption,
    NumericOption,
    StringOption,
    IntegerOption,
    FileOption,
    TupleOption
)


class FalsecolorOptions(OptionCollection):
    """falsecolor options.

    Also see: https://floyd.lbl.gov/radiance/man_html/falsecolor.1.html
    """

    __slots__ = (
        "_pal",
        "_s",
        "_m",
        "_log",
        "_l",
        "_cl",
        "_p",
        "_cb",
        "_n",
        "_lw",
        "_lh",
        "_e",
        "_r",
        "_g",
        "_b",
        "_odim"
    )

    def __init__(self):
        """falsecolor command options."""

        OptionCollection.__init__(self)
        self._pal = StringOption(
            "pal", "Color palette - default: def",
            valid_values=["def", "spec", "hot", "pm3d"], whole=True
        )
        self._s = StringOption("s", "Legend scale - default: auto")
        self._m = NumericOption("m", "Legend multiplier - default: 179")
        self._log = IntegerOption("log", "Logarithmic mapping intervals")
        self._l = StringOption("l", "Legend label - default: Nits")
        self._cl = BoolOption("cl", "Contour lines - default: False")
        self._p = FileOption("p", "Picture to place countour lines over")
        self._cb = BoolOption("cb", "Contour bands - default: False")
        self._n = IntegerOption("n", "Number of contours - default: 8")
        self._lw = IntegerOption("lw", "Legend width in pixels - default: 100")
        self._lh = IntegerOption("lh", "Legend width in pixels - default: 200")
        self._e = BoolOption("e", "Print extrema min/max pixels - default: False")
        self._r = StringOption("r", "Red channel mapping (expression of 'v')")
        self._g = StringOption("g", "Green channel mapping (expression of 'v')")
        self._b = StringOption("b", "Blue channel mapping (expression of 'v')")
        self._odim = TupleOption(
            "odim", "X and Y grid dimensions for value overlay", length=2,
            numtype=int
        )
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection.
        """
        if self._cl.is_set and self._cb.is_set:
            raise ValueError(
                'Both cl and cb are set. The program can use either draw contour '
                'lines or contour bands, but not both.'
            )

    @property
    def pal(self):
        """Image and legend color palette - default: def

        *   'def' sets default colors.
        *   'pm3d' sets a variation of the default colors.
        *   'spec' sets the old spectral mapping.
        *   'hot' sets a thermal scale.
        """
        return self._pal

    @pal.setter
    def pal(self, value):
        self._pal.value = value

    @property
    def s(self):
        """Legend scale - default: auto

        If the argument given begins with an "a" for "auto," then the maximum
        is used for scaling the result.
        """
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def m(self):
        """Legend multiplier (integer) - default: 179

        The default multiplier is 179, which converts from radiance or irradiance
        to luminance or illuminance, respectively.
        """
        return self._m

    @m.setter
    def m(self, value):
        self._m.value = value

    @property
    def log(self):
        """Number of decades to use with a logarithmic legend scale

        Decades are the number of intervale of 10 below the maximum scale.
        If unspecified, a linear scale is used.
        """
        return self._log

    @log.setter
    def log(self, value):
        self._log.value = value

    @property
    def l(self):
        """Text for the legend label - default: Nits

        The default label of "Nits" is appropriate for standard Radiance images.
        If the -i option of rpict was used to produce the image, then the
        appropriate label would be "Lux".
        """
        return self._l

    @l.setter
    def l(self, value):
        self._l.value = value

    @property
    def cl(self):
        """Draw contour lines - default: False

        Note that contour lines are used instead of creating a falsecolor image.
        """
        return self._cl

    @cl.setter
    def cl(self, value):
        self._cl.value = value

    @property
    def p(self):
        """Picture file path over which contour lines will be placed
        """
        return self._p

    @p.setter
    def p(self, value):
        self._p.value = value

    @property
    def cb(self):
        """Draw contour bands - default: False

        The thickness of the bands is related to the rate of change in the image.
        Note that contour bands are used instead of creating a falsecolor image.
        """
        return self._cb

    @cb.setter
    def cb(self, value):
        self._cb.value = value

    @property
    def n(self):
        """Number of contours - default: 8

        Note that this also changes the corresponding legend entries.
        """
        return self._n

    @n.setter
    def n(self, value):
        self._n.value = value

    @property
    def lw(self):
        """Legend width in pixels - default: 100

        A value of zero in either eliminates the legend in the output.
        """
        return self._lw

    @lw.setter
    def lw(self, value):
        self._lw.value = value

    @property
    def lh(self):
        """Legend height in pixels - default: 200

        A value of zero in either eliminates the legend in the output.
        """
        return self._lh

    @lh.setter
    def lh(self, value):
        self._lh.value = value

    @property
    def e(self):
        """Print extrema min/max pixels - default: False

        Extrema are the brightest and darkest pixels of the input picture.
        """
        return self._e

    @e.setter
    def e(self, value):
        self._e.value = value

    @property
    def r(self):
        """Red channel mapping as an expression of 'v'

        The expression must be of the variable v, where v varies from 0 to 1.
        This option is not recommended for the casual user.
        """
        return self._r

    @r.setter
    def r(self, value):
        self._r.value = value

    @property
    def g(self):
        """Green channel mapping as an expression of 'v'

        The expression must be of the variable v, where v varies from 0 to 1.
        This option is not recommended for the casual user.
        """
        return self._g

    @g.setter
    def g(self, value):
        self._g.value = value

    @property
    def b(self):
        """Blue channel mapping as an expression of 'v'

        The expression must be of the variable v, where v varies from 0 to 1.
        This option is not recommended for the casual user.
        """
        return self._b

    @b.setter
    def b(self, value):
        self._b.value = value

    @property
    def odim(self):
        """"X and Y grid dimensions for value overlay.

        The expression must be a tuple with two integer values.
        """
        return self._odim

    @odim.setter
    def odim(self, value):
        self._odim.value = value
