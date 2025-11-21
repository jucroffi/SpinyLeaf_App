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


class EvalglareOptions(OptionCollection):
    """evalglare options.

    Also see: https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/evalglare.pdf/view
    """

    __slots__ = (
        "_A",
        "_B",
        "_b",
        "_c",
        "_C",
        "_d",
        "_f",
        "_g",
        "_G",
        "_i",
        "_I",
        "_l",
        "_L",
        "_N",
        "_q",
        "_r",
        "_s",
        "_t",
        "_T",
        "_u",
        "_x",
        "_y",
        "_Y"
    )

    def __init__(self):
        """evalglare command options."""

        OptionCollection.__init__(self)
        self._A = FileOption("A", "Masking file to study a certain area")
        self._B = NumericOption("B", "Angle to calculate luminance of horizontal band")
        self._b = NumericOption("b", "Threshold factor in cd/m2 - default: 2000")
        self._c = FileOption("c", "Output check file path")
        self._C = StringOption(
            "C", "Correction mode - default l+",
            valid_values=["0", "l+", "l-"], whole=True
        )
        self._d = BoolOption("d", "Enable detailed output - default: False")
        self._f = BoolOption("f", "Forcing option for -vtv and black corners")
        self._g = IntegerOption(
            "g", "Cut field of view according to Guth; no eval", min_value=1, max_value=2
        )
        self._G = IntegerOption(
            "G", "Cut field of view according to Guth; eval", min_value=1, max_value=2
        )
        self._i = NumericOption("i", "Externally measured vertical illuminance in lux")
        self._I = TupleOption(
            "I", "Externally measured illuminance as (Ev, y_max, y_min)",
            value=None, length=3, numtype=float
        )
        self._l = TupleOption(
            "l", "Circular one zone evaluation as (xpos, ypos, angle)",
            value=None, length=3, numtype=float
        )
        self._L = TupleOption(
            "L", "Circular two zone evaluation as (xpos, ypos, angle1, angle2)",
            value=None, length=4, numtype=float
        )
        self._N = TupleOption(
            "N", "Pixel replacement during overflow as (xpos, ypos, angle, Ev, fname)",
            value=None, length=5, numtype=str
        )
        self._q = IntegerOption(
            "q", "Background luminance calculation method - default: 0",
            min_value=0, max_value=2
        )
        self._r = NumericOption(
            "r", "Search radius (angle) between pixels - default: 0.2 radians"
        )
        self._s = BoolOption("s", "Enable smoothing function - default: False")
        self._t = TupleOption(
            "t", "Task position as (xpos, ypos, angle)",
            value=None, length=3, numtype=float
        )
        self._T = TupleOption(
            "T", "Task position (colored blue) as (xpos, ypos, angle)",
            value=None, length=3, numtype=float
        )
        self._u = TupleOption(
            "u", "RGB color to color glare sources uniformly",
            value=None, length=3, numtype=float
        )
        self._x = BoolOption("x", "Disable peak extraction - default: False")
        self._y = BoolOption("y", "Enable peak extraction - default: True")
        self._Y = NumericOption("Y", "Enable peak extraction with value in cd/m2")
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.
        """
        warn = ' This program can use either of the options but not both.'
        if self._g.is_set and self._G.is_set:
            raise ValueError('Both -g and -G do not go well together.' + warn)
        if self._i.is_set and self._I.is_set:
            raise ValueError('Both -i and -I do not go well together.' + warn)
        if self._l.is_set and self._L.is_set:
            raise ValueError('Both -l and -L do not go well together.' + warn)
        if self._t.is_set and self._T.is_set:
            raise ValueError('Both -t and -T do not go well together.' + warn)
        if self._y.is_set and self._Y.is_set:
            raise ValueError('Both -y and -Y do not go well together.' + warn)
        if self._A.is_set and (self._l.is_set or self._L.is_set):
            raise ValueError('Both -A and -l/-L do not go well together.')

    @property
    def A(self):
        """Masking file to study a certain area

        Use a masking file to enable statistical analysis on a certain area.
        Does not affect glare source detection. The masking threshold is 0.1 cd/m2
        (all pixels exceeding 0.1 cd/m2 are treated as "inside" the mask.) The PGSV
        equations are also calculated, but require the masking area to be the window.
        It requires the -d option. The result of the analysis of the masking area
        is given in the first line of the output. Should not be combined with
        -l or -L options.
        """
        return self._A

    @A.setter
    def A(self, value):
        self._A.value = value

    @property
    def B(self):
        """Angle to calculate luminance of horizontal band

        Calculate average luminance of a horizontal band. The angle is in radians.
        This calculation does not affect glare source detection. Output only when
        using the -d option. The result of the analysis of the band is given in
        the first line of the output.
        """
        return self._B

    @B.setter
    def B(self, value):
        self._B.value = value

    @property
    def b(self):
        """Threshold factor in cd/m2 - default: 2000

        Threshold factor; if factor is larger than 100, it is used as constant
        threshold in cd/m2, regardless if a task position is given or not. If
        factor is less or equal than 100 and a task position is given, this factor
        multiplied by the average task luminance will be used as threshold for
        detecting the glare sources. If factor is less than or equal to 100 and
        no task position is given, this factor multiplied by the average luminance in
        the entire picture will be used as threshold for detecting the glare
        sources (not recommended). Default value of factor is 2000 (fixed
        threshold method).
        """
        return self._b

    @b.setter
    def b(self, value):
        self._b.value = value

    @property
    def c(self):
        """Output check file path

        Checkfile is written in the RADIANCE picture format.
        """
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value

    @property
    def C(self):
        """Correction mode - default l+

        Type 0: all corrections turned off , Type l+: Low light correction applied
        (default), Type l-: Low light correction disabled.
        """
        return self._C

    @C.setter
    def C(self, value):
        self._C.value = value

    @property
    def d(self):
        """Enable detailed output - default: False"""
        return self._d

    @d.setter
    def d(self, value):
        self._d.value = value

    @property
    def f(self):
        """Forcing option for -vtv and black corners - default: False
        
        This forcing option prevents from stopping when -vtv is used and black
        corners are detected.
        """
        return self._f

    @f.setter
    def f(self, value):
        self._f.value = value

    @property
    def g(self):
        """Cut field of view according to Guth with no glare evaluation
        
        Cut field of view according to Guth, write checkfile specified by -c and
        exit without any glare evaluation. Type 1: total field of view. Type 2: field
        of view seen by both eyes
        """
        return self._g

    @g.setter
    def g(self, value):
        self._g.value = value

    @property
    def G(self):
        """Cut field of view according to Guth and perform glare evaluation
        
        Type 1: total field of view. Type 2: field of view seen by both eyes
        """
        return self._G

    @G.setter
    def G(self, value):
        self._G.value = value

    @property
    def i(self):
        """Externally measured vertical illuminance in lux
        
        The vertical illuminance Ev in lux is measured externally. This value will
        be used for calculating the DGP.
        """
        return self._i

    @i.setter
    def i(self, value):
        self._i.value = value

    @property
    def I(self):
        """Externally measured illuminance as (Ev, y_max, y_min)
        
        The vertical illuminance Ev in lux is measured externally. This value will
        be used for calculating the DGP. Below y_min and above y_max, the picture
        is filled up by the last known value. This option should be used, when
        the provided picture is cut horizontally.
        """
        return self._I

    @I.setter
    def I(self, value):
        self._I.value = value

    @property
    def l(self):
        """Circular one zone evaluation as (xpos, ypos, angle)
        
        Activate circular one zone evaluation. The center of the zone is given by
        xpos and ypos. The opening angle of the zone is specified in radians.
        The result of the analysis of zone1 is given in the first line of the output.
        """
        return self._l

    @l.setter
    def l(self, value):
        self._l.value = value

    @property
    def L(self):
        """Circular two zone evaluation as (xpos, ypos, angle1, angle2)
        
        Activate circular two zone evaluation. The center of the zone is given by
        xpos and ypos. The opening angle of the inner zone1 is specified by angle1
        in radians, the opening angle of the outer zone2 by angle2. The result of
        the analysis of the zones is given in the first two lines of the output.
        """
        return self._L

    @L.setter
    def L(self, value):
        self._L.value = value

    @property
    def N(self):
        """Pixel replacement during overflow as (xpos, ypos, angle, Ev, fname)
        
        Pixel replacement in case of pixel overflow in hdr image and measured Ev
        (in lux) is available. Writes the modified image to fname and exists
        immediately (without glare evaluation). Replaces pixels in a circular
        zone to match Ev. The center of the zone is given by xpos and ypos. The
        opening angle of the zone is specified in radians.
        
        This option should be applied very carefully and only exceptionally.
        Pixel overflow should be avoided from the beginning by applying shorter
        exposure times and/or neutral filters.
        """
        return self._N

    @N.setter
    def N(self, value):
        self._N.value = value

    @property
    def q(self):
        """Background luminance calculation method - default: 0

        Toggle modes for the background luminance calculation.

        * 0 (default) - CIE-mode Lb = (Ev - Edir) / pi
        * 1 - Lb = mathematical average luminance without glare sources
        * 2 (not recommended) - Lb = Ev / pi
        """
        return self._q

    @q.setter
    def q(self, value):
        self._q.value = value

    @property
    def r(self):
        """Search radius (angle) between pixels - default: 0.2 radians

        Search radius (angle in radians) between pixels, where evalglare tries to merge
        glare source pixels to the same glare source (default value: 0.2 radians)
        """
        return self._r

    @r.setter
    def r(self, value):
        self._r.value = value

    @property
    def s(self):
        """Enable smoothing function - default: False"""
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def t(self):
        """Task position as (xpos, ypos, angle)

        Definition of task position in x and y coordinates, and its opening
        angle in radians.
        """
        return self._t

    @t.setter
    def t(self, value):
        self._t.value = value

    @property
    def T(self):
        """Task position as (xpos, ypos, angle)

        Same as -t, except that the task area is colored bluish in the checkfile.
        """
        return self._T

    @T.setter
    def T(self, value):
        self._T.value = value

    @property
    def u(self):
        """RGB color to color glare sources uniformly

        Color glare sources uniformly when writing check file (implies -c option).
        Color given in r g b. (in any range, values are normalized)
        """
        return self._u

    @u.setter
    def u(self, value):
        self._u.value = value

    @property
    def x(self):
        """Disable peak extraction - default: False"""
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value

    @property
    def y(self):
        """Enable peak extraction - default: True"""
        return self._y

    @y.setter
    def y(self, value):
        self._y.value = value

    @property
    def Y(self):
        """Enable peak extraction with value in cd/m2

        Enable peak extraction with value (in cd/m2) as threshold for extracted peaks.
        """
        return self._Y

    @Y.setter
    def Y(self, value):
        self._Y.value = value
