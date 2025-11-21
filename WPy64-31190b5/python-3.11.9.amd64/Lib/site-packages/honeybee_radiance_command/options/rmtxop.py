"""Rmtxop parameters"""
from .optionbase import OptionCollection, BoolOption, StringOptionJoined


class RmtxopOptions(OptionCollection):
    """
    [-v] [-f[afdc]]

    Also see: https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/rmtxop.pdf
    """

    __slots__ = ('_v', '_f',)

    def __init__(self):
        """rmtxop command options."""
        OptionCollection.__init__(self)
        self._on_setattr_check = False
        self._v = BoolOption('v', 'Turn on verbose reporting. Defaults to False')
        self._f = StringOptionJoined('f', 'output data format. Set "a" for ASCII, "d" for'
                                          ' binary doubles, "f" floats and "c" for RGBE'
                                          'colors', valid_values=('a', 'f', 'd', 'c'))

    @property
    def v(self):
        """Turn on verbose reporting, which announces each matrix operation"""
        return self._v

    @v.setter
    def v(self, value):
        self._v.value = value

    @property
    def f(self):
        """Output data format. Can be specified as ASCII(a), binary double(d),
        float(f) or RGBE colors(c). By default, the output is written in the lowest
        resolution format among inputs."""
        return self._f

    @f.setter
    def f(self, value):
        self._f.value = value
