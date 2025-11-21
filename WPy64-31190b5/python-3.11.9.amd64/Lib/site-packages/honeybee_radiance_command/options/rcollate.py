"""Rcollate parameters."""

from .optionbase import OptionCollection, StringOptionJoined, StringOption, \
    IntegerOption, BoolOption


class RcollateOptions(OptionCollection):
    """Rcollate command options.

    [ -h[io] ][ -w ][ -f[afdb][N] ][ -t ][ -ic in_col ][ -ir in_row ][ -oc out_col ]
    [ -or out_row ][ -o RxC[xR1xC2..] ]

    In this class the -f option has been implemented such that there are separate options
    like -fa , -ff etc. This is to enable the inclusion of numeric inputs for digits
    in an easy manner.

    Also see: https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/rcollate.pdf
    """
    __slots__ = ('_h', '_w', '_fa', '_ff', '_fd', '_fb', '_t', '_ic', '_ir', '_oc',
                 '_or', '_o')

    def __init__(self):
        """rcollate command options"""
        OptionCollection.__init__(self)
        self._on_setattr_check = False
        self._h = StringOptionJoined('h', 'Header availability.',
                                     valid_values=('', 'i', 'o'))
        self._w = BoolOption('w', 'Turn off non-fatal warning messages')
        self._fa = StringOptionJoined('fa', 'Number of space-separated words/numbers in '
                                            ' each record assumed to be in ASCII format.')
        self._ff = StringOptionJoined('ff', 'Number of words in the input,'
                                            ' each record assumed to be in '
                                            'floating-point format.')
        self._fd = StringOptionJoined('fd', 'Number of words in the input,'
                                            ' each record assumed to be in double '
                                            'format.')
        self._fb = StringOptionJoined('fb', 'Number of words in the input,'
                                            ' each record assumed to be in binary '
                                            'format.')
        self._t = BoolOption('t', 'Swap rows and columns in the input.')
        self._ic = IntegerOption('ic', 'Input columns')
        self._ir = IntegerOption('ir', 'Input rows')
        self._oc = IntegerOption('oc', 'Output columns')
        # or  is a protected keyword, so using oR
        self._or = IntegerOption('or', 'Output rows')
        self._o = StringOption('o', 'Output matrix shape.')
        self._on_setattr_check = True

    @property
    def h(self):
        """By default, header is expected to be present. "i" option turns off the
        expectation for input header and "o" option for output header Providing an
        empty string turns off both headers. If just an empty string is provided then
        both headers will be assumed to be empty."""
        return self._h

    @h.setter
    def h(self, value):
        self._h.value = value

    @property
    def w(self):
        """The -w option turns off non-fatal warning messages, such as unexpected EOD."""
        return self._w

    @w.setter
    def w(self, value):
        self._w.value = value

    @property
    def fa(self):
        """Number of words in the input, each record assumed to be in
        ASCII format. If an empty string is provided as input, then the number will be
        determined by Radiance while parsing."""
        return self._fa

    @fa.setter
    def fa(self, value):
        self._fa.value = value

    @property
    def ff(self):
        """Number of words in the input, each record assumed to be in floating-point
        format. If an empty string is provided as input, then the number will be
        determined by Radiance while parsing."""
        return self._ff

    @ff.setter
    def ff(self, value):
        self._ff.value = value

    @property
    def fd(self):
        """Number of words in the input, each record assumed to be in double format. If
        an empty string is provided as input, then the number will be determined by
        Radiance while parsing."""
        return self._fd

    @fd.setter
    def fd(self, value):
        self._fd.value = value

    @property
    def fb(self):
        """Number of words in the input, each record assumed to be in binary format. If
        an empty string is provided as input, then the number will be determined by
        Radiance while parsing."""
        return self._fb

    @fb.setter
    def fb(self, value):
        self._fb.value = value

    @property
    def t(self):
        """The transpose option, -t swaps rows and columns on the input."""
        return self._t

    @t.setter
    def t(self, value):
        self._t.value = value

    @property
    def ic(self):
        """Input columns"""
        return self._ic

    @ic.setter
    def ic(self, value):
        self._ic.value = value

    @property
    def ir(self):
        """Input rows"""
        return self._ir

    @ir.setter
    def ir(self, value):
        self._ir.value = value

    @property
    def oc(self):
        """Output columns"""
        return self._oc

    @oc.setter
    def oc(self, value):
        self._oc.value = value

    @property
    def or_(self):
        """Output rows"""
        return self._or

    @or_.setter
    def or_(self, value):
        self._or.value = value

    @property
    def o(self):
        """The number of rows may be specified with a -or option, or may be determined
        automatically from the size of the input if it is an even multiple of the
        number of columns (as it should be). Alternatively, both may be specified
        using a -o option with the number of rows and columns separated by an x,
        as in "30x14" for 30 rows by 14 columns. Rcollate can also reorder the input
        into nested blocks by continuing the output size string. For example,
        "3x10X7x2" would order output data with a 3x10 super-array of 7x2 subblocks.
        This type of block hierarchy is convenient for visualizing tensor data.
        If the -o option is also given with multiple block levels, the transpose
        operation will logically precede the reordering operation, regardless of their
        position on the command line.
        """
        return self._o

    @o.setter
    def o(self, value):
        err_msg = 'The specified value for of "o" is %s.' % value
        err_msg += 'The value for "o" must be specified with the number of rows and ' \
                   'columns separated by an x, as in "30x14" for 30 rows by 14 columns. ' \
                   'Rcollate can also reorder the input into nested blocks by continuing ' \
                   'the output size string. For example, "3x10X7x2" would order output ' \
                   'data with a 3x10 super-array of 7x2 subblocks. '

        if not 'x' in value:
            raise ValueError(err_msg)

        try:
            int_test = [int(num) for num in value.split('x')]
        except ValueError:
            err_msg += 'The integers required for specifying the shape of the output ' \
                       'matrix are not correctly provided. '
            raise ValueError(err_msg)

        self._o.value = value

    def _on_setattr(self):
        """This method executes after setting each new attribute.
        """
        all_formats = [1 for i in (self._fa, self._fb, self._fd, self._ff) if i.is_set]
        if sum(all_formats) > 1:
            raise ValueError(
                'Only one of the -fa, -fb, -fd, -ff options can be set at a time.')

        output_shape_set = self._o.is_set and (self._oc.is_set or self._or.is_set)
        if output_shape_set:
            raise ValueError('The options for setting the output shape .i.e can either '
                             'be set through o or (or and oc). They cannot be set at the'
                             ' same time')
