"""Dctimestep parameters"""
from .optionbase import OptionCollection, StringOption, IntegerOption, BoolOption, \
    StringOptionJoined


class DctimestepOptions(OptionCollection):
    """
    [-n nsteps] [-h][-o ospec][-i{f|d}][-o{f|d|c}]

    Also see: https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/dctimestep.pdf
    """

    __slots__ = ('_n', '_h', '_o', '_i', '_op_fmt')

    def __init__(self):
        """dctimestep command options."""
        OptionCollection.__init__(self)
        self._h = BoolOption('h', 'remove header in output file. default:header is '
                                  'written')
        self._i = StringOptionJoined('i', 'input data format for the sky vector. "f" '
                                          'indicates float and "d" indicates double. '
                                          'Double is efficient for large matrices. This'
                                          'option is unnecessary if the sky vector/matrix '
                                          'includes a header', valid_values=['d', 'f'],
                                     whole=False)
        self._o = StringOption('o', 'The -o option may be used to specify a file or '
                                    'a set of output files to use rather than the '
                                    'standard output. If the given specification '
                                    'contains a "%d" format string, this will be '
                                    'replaced by the time step index, starting from 0.'
                                    'In this way, multiple output pictures or '
                                    'separate result vectors may be produced.')
        self._op_fmt = StringOptionJoined('o', 'The -of, -od or -oc option may be used to'
                                          'specify IEEE float, double or RGBE (picture) '
                                          'output data, respectively.',
                                          valid_values=['f', 'd', 'c'])
        self._n = IntegerOption('n',
                                'The -n option may be used to indicate the number of '
                                'time steps, which will be 1 for a sky vector. This '
                                'option is unnecessary if the sky vector/matrix '
                                'includes a header')

    @property
    def h(self):
        """Remove header."""
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def o(self):
        """Output file or output files format string"""
        return self._o

    @o.setter
    def o(self, value):
        self._o.value = value

    @property
    def op_fmt(self):
        """output data format"""
        return self._op_fmt

    @op_fmt.setter
    def op_fmt(self, value):
        self._op_fmt.value = value

    @property
    def i(self):
        """Input data format for the sky vector"""
        return self._i

    @i.setter
    def i(self, value):
        self._i.value = value

    @property
    def n(self):
        """Number of time steps"""
        return self._n

    @n.setter
    def n(self, value):
        self._n.value = value
