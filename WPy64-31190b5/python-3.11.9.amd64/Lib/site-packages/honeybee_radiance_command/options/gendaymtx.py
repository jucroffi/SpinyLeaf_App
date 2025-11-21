"""Gendaymtx parameters."""
from .optionbase import NumericOption, OptionCollection, FileOption, \
    StringOptionJoined, TupleOption, IntegerOption, BoolOption
import honeybee_radiance_command._exception as exceptions


class GendaymtxOptions(OptionCollection):
    """
    [-v][-h][-A][-d|-s|-n][-u][-D file [-M modfile]][-r deg][-m N][-g r g b][-c r g b]
    [-o{f|d}][-O{0|1}]

    Also see: https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/gendaymtx.pdf
    """

    __slots__ = (
        '_v', '_h', '_A', '_d', '_s', '_n', '_u', '_D', '_M', '_r', '_m', '_g', '_c',
        '_o', '_O'
    )

    def __init__(self):
        """gendaymtx command options."""
        OptionCollection.__init__(self)
        self._on_setattr_check = False
        self._v = BoolOption('v', 'verbose reporting - default: off')
        self._h = BoolOption(
            'h', 'prevents the output of the usual header information - default: off'
        )
        self._A = BoolOption(
            'A', 'tells gendaymtx to generate a single column corresponding to an '
            'average sky computed over all the input time steps, rather than one '
            'column per time step. - default: off'
        )
        self._m = IntegerOption('m', 'Set sky resolution - default: 1', min_value=1)
        self._d = BoolOption(
            'd', 'produce a sun-only matrix, with no sky contributions, and the ground '
            'patch also set to zero.'
        )
        self._s = BoolOption(
            's', 'exclude any direct solar component from the output, with the rest of '
            'the sky and ground patch unaffected.'
        )
        self._n = BoolOption(
            'n', 'may be used if no matrix output is desired at all. This may be used '
            'to merely check the input, or in combination with the -D option.'
        )
        self._u = BoolOption(
            'u', 'ignores input times when the sun is below the horizon. This is a '
            'convenient way to average daylight hours only with the -A option or to '
            'ensure that matrix entries correspond to solar positions produced with '
            'the -D option'
        )
        self._c = TupleOption(
            'c', 'may be used to specify a color for the sky. The gray value should '
            'equal 1 for proper energy balance. The default sky color is -c 0.960 '
            '1.004 1.118', length=3, numtype=float
        )
        self._g = TupleOption(
            'g', 'may be used to specify a ground color. The default value is -g 0.2 '
            '0.2 0.2 corresponding to a 20 percent gray', length=3, numtype=float
        )
        self._D = FileOption(
            'D', 'may be used to specify an output file to contain a list of solar '
            'positions and intensities corresponding to time steps in the weather tape '
            'where the sun has any portion above the horizon. Sun radiance values may '
            'be zero if the direct amount is zero on the input. Sun modifiers and names '
            'will be indexed by the minute, numbered from midnight, January 1st.'
        )
        self._M = FileOption(
            'M', 'may be used to specify an output file to contain a list of sun '
            'modifiers.'
        )
        self._r = NumericOption(
            'r', 'rotates the sky the specified number of degrees counter-clockwise'
            'about the zenith, i.e., west of north.', min_value=-360, max_value=360
        )
        self._o = StringOptionJoined(
            'o', 'The -of or -od option may be used to specify binary float or double '
            'output, respectively. This is much faster to write and to read, and is '
            'therefore preferred on systems that support it.', valid_values=['f', 'd'],
            whole=False
        )
        self._O = StringOptionJoined(
            'O', 'The -O1 option specifies that output should be total solar radiance '
            'rather than visible radiance - default O0', valid_values=['0', '1'],
            whole=False
        )
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute."""
        if self.s.is_set and self.d.is_set:
            raise exceptions.ExclusiveOptionsError(self.command, 's', 'd')
        if self.s.is_set and self.n.is_set:
            raise exceptions.ExclusiveOptionsError(self.command, 's', 'n')
        if self.n.is_set and self.d.is_set:
            raise exceptions.ExclusiveOptionsError(self.command, 'n', 'd')

    @property
    def v(self):
        """Verbose reporting."""
        return self._v

    @v.setter
    def v(self, value):
        self._v.value = value

    @property
    def h(self):
        """Remove header."""
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def A(self):
        """Average sky calculated over all the timesteps."""
        return self._A

    @A.setter
    def A(self, value):
        self._A.value = value

    @property
    def m(self):
        """Sky resolution -- default: 1"""
        return self._m

    @m.setter
    def m(self, value):
        self._m.value = value

    @property
    def d(self):
        """Produce a sun only matrix."""
        return self._d

    @d.setter
    def d(self, value):
        self._d.value = value

    @property
    def s(self):
        """Exclude direct solar."""
        return self._s

    @s.setter
    def s(self, value):
        self._s.value = value

    @property
    def n(self):
        """No output to stdout."""
        return self._n

    @n.setter
    def n(self, value):
        self._n.value = value

    @property
    def u(self):
        """A boolean for sun up hours."""
        return self._u

    @u.setter
    def u(self, value):
        self._u.value = value

    @property
    def c(self):
        """Sky color."""
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value

    @property
    def g(self):
        """Ground color."""
        return self._g

    @g.setter
    def g(self, value):
        self._g.value = value

    @property
    def D(self):
        """Output file for list of solar positions."""
        return self._D

    @D.setter
    def D(self, value):
        self._D.value = value

    @property
    def M(self):
        """Output file for list of solar modifiers."""
        return self._M

    @M.setter
    def M(self, value):
        self._M.value = value

    @property
    def r(self):
        """Rotation from north."""
        return self._r

    @r.setter
    def r(self, value):
        self._r.value = value

    @property
    def o(self):
        """Output type."""
        return self._o

    @o.setter
    def o(self, value):
        self._o.value = value

    @property
    def O(self):
        """Output format -- default: 0

        The -O1 option specifies that output should be total solar radiance rather than '
        'visible radiance - default O0
        """
        return self._O

    @O.setter
    def O(self, value):
        self._O.value = value
