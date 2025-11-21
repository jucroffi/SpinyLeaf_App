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


class PcondOptions(OptionCollection):
    """pcond options.

    Also see: https://floyd.lbl.gov/radiance/man_html/pcond.1.html
    """

    __slots__ = (
        "_h",
        "_a",
        "_v",
        "_s",
        "_c",
        "_w",
        "_i",
        "_I",
        "_l",
        "_e",
        "_u",
        "_d",
        "_p",
        "_f",
        "_x"
    )

    def __init__(self):
        """pcond command options."""

        OptionCollection.__init__(self)
        self._h = BoolOption("h", "Human visual response - default: False")
        self._a = BoolOption("a", "Human visual acuity loss - default: False")
        self._v = BoolOption("v", "Veiling glare - default: False")
        self._s = BoolOption("s", "Human contrast sensitivity - default: False")
        self._c = BoolOption("c", "Color visibility loss - default: False")
        self._w = BoolOption("w", "Weighted average exposure - default: False")
        self._i = IntegerOption("i", "Importance of fixation points -default: 0")
        self._I = BoolOption("I", "Precomputed histogram - default: False")
        self._l = BoolOption("l", "Linear response function - default: False")
        self._e = StringOption("e", "Exposure adjustment")
        self._u = IntegerOption("u", "Top of Luminance - default: 100")
        self._d = NumericOption("d", "Dynamic range - default: 32")
        self._p = TupleOption("p", "RGB primaries", value=None, length=8, numtype=float)
        self._f = FileOption("f", "output file from macbethcal")
        self._x = FileOption("x", "Output display luminance to mapfile")
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection. For
        instance in pcond option collection -f and -p don't go together very well.
        You can include a check to ensure this is always correct.
        """

        if self._f.is_set and self._p.is_set:
            raise ValueError(
                'Both -f and -p do not go well together.'
                ' This program can use either of the options but not both.'
                    )

    @property
    def h(self):
        """Human visual response - default: False

        Mimic human visual response in the output. The goal of this process is to
        produce output that correlates strongly with a person's subjective
        impression of a scene. This switch is a bundle of the −a, −v, −s and −c options.
        """
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def a(self):
        """Human visual acuity loss - default: False

        Defocus darker regions of the image to simulate human visual acuity loss.
        This option will not affect well-lit scenes.
        """
        return self._a

    @a.setter
    def a(self, value):
        self._a.value = value

    @property
    def v(self):
        """Veiling glare - default: False

        Add veiling glare due to very bright regions in the image. This simulates
        internal scattering in the human eye, which results in a loss of visible
        contrast near bright sources.
        """
        return self._v

    @v.setter
    def v(self, value):
        self._v.value = value

    @property
    def s(self):
        """Human contrast sensitivity - default: False

        Use the human contrast sensitivity function in determining the exposure for 
        the image. A darker scene will have relatively lower exposure with lower
        contrast than a well-lit scene.
        """
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def c(self):
        """Color visibility loss - default: False

        If parts of the image are in the mesopic or scotopic range where the cone
        photoreceptors lose their efficiency, this switch will cause a corresponding
        loss of color visibility in the output and a shift to a scotopic (blue-dominant)
        response function.
        """
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value

    @property
    def w(self):
        """Weighted average exposure - default: False

        Use a center-weighted average for the exposure rather than the default uniform
        average. This may improve the exposure for scenes with high or low peripheral
        brightness.
        """
        return self._w

    @w.setter
    def w(self, value):
        self._w.value = value

    @property
    def i(self):
        """Importance of fixation points -default: 0

        Set the relative importance of fixation points to a value, which is a value
        between 0 and 1. If fixfrac is zero (the default), then no fixation points
        are used in determining the local or global adaptation. If the value is greater
        than zero, then a list of fixation points is read from the standard input.
        These points are given as tab-separated (x,y) picture coordinates, such as
        those produced by the −op option of ximage(1). The foveal samples about these
        fixation points will then be weighted together with the global averaging scheme
        such that the fixations receive the value of the total weight. If the value is
        one, then only the fixation points are considered for adaptation.
        """
        return self._i

    @i.setter
    def i(self, value):
        self._i.value = value

    @property
    def I(self):
        """Precomputed histogram - default: False

        Rather than computing a histogram of foveal samples from the source picture,
        use the precomputed histogram provided on the standard input. This data should
        be given in pairs of the base-10 logarithm of world luminance and a count for
        each bin in ascending order, as computed by the phisto(1) script. This option
        is useful for producing identical exposures of multiple pictures
        (as in an animation), and provides greater control over the histogram
        computation.
        """
        return self._I

    @I.setter
    def I(self, value):
        self._I.value = value

    @property
    def l(self):
        """Linear response function - default: False

        Use a linear response function rather than the standard dynamic range
        compression algorithm. This will prevent the loss of usable physical
        values in the output picture, although some parts of the resulting image may
        be too dark or too bright to see.
        """
        return self._l

    @l.setter
    def l(self, value):
        self._l.value = value

    @property
    def e(self):
        """Exposure adjustment

        Set the exposure adjustment for the picture to a value. This may either be a
        real multiplier, or a (fractional) number of f-stops preceeded by
        a + or -. This option implies a linear response (see the −l option above).
        """
        return self._e

    @e.setter
    def e(self, value):
        self._e.value = value

    @property
    def u(self):
        """Top of Luminance - default: 100

        Specifies the top of the luminance range for the target output device.
        That is, the luminance (in candelas/m^2) for an output pixel value of
        (R,G,B)=(1,1,1). The default value is 100 cd/m^2.
        """
        return self._u

    @u.setter
    def u(self, value):
        self._u.value = value

    @property
    def d(self):
        """Dynamic range - default: 32

        Specifies the dynamic range for the target output device, which is the
        ratio of the maximum and minimum usable display luminances. The default
        value is 32.
        """
        return self._d

    @d.setter
    def d(self, value):
        self._d.value = value

    @property
    def p(self):
        """RGB primaries

        Specifies the RGB primaries for the target output device.
        These are the 1931 CIE (x,y) chromaticity values for red, green, blue and
        white, respectively in the format of (xr yr xg yg xb yb xw yw).
        """
        return self._p

    @p.setter
    def p(self, value):
        self._p.value = value

    @property
    def f(self):
        """output file from macbethcal

        Use the given output file from macbethcal(1) to precorrect the color and
        contrast for the target output device. This does a more thorough job than
        a simple primary correction using the −p option. Only one of −f or −p may
        be given.
        """
        return self._f

    @f.setter
    def f(self, value):
        self._f.value = value

    @property
    def x(self):
        """Output display luminance to mapfile

        Put out the final mapping from world luminance to display luminance to mapfile.
        This file will contain values from the minimum usable world luminance to
        the maximum (in candelas/m^2) in one column, and their corresponding display
        luminance values (also in candelas/m^2) in the second column. This file may
        be used for debugging purposes, or to plot the mapping function created by
        pcond.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value
    
