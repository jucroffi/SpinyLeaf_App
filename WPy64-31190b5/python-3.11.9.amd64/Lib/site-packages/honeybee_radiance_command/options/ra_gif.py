"""ra_gif parameters."""
import math
from .optionbase import OptionCollection, BoolOption, NumericOption, IntegerOption


class Ra_GIFOptions(OptionCollection):
    """ra_gif command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/ra_gif.1.html
    """
    __slots__ = ('_b', '_d', '_c', '_g', '_e', '_n')

    def __init__(self):
        """ra_gif command options."""

        OptionCollection.__init__(self)
        self._b = BoolOption("b", "Change image color - default: False")
        self._d = BoolOption("d", "Turn off dithering - default: False")
        self._c = NumericOption("c", "Fewer colors", min_value=1, max_value=256)
        self._g = NumericOption("g", "Gamma correction - default: 2.2")
        self._e = IntegerOption("e", "Exposure compensation")
        self._n = NumericOption("n", "Sampling factor for large images", min_value=1,
                                max_value=80)

    @property
    def b(self):
        """Change image color - default: False

        Convert a radiance generated image to black and white.
        """
        return self._b

    @b.setter
    def b(self, value):
        self.b.value = value

    @property
    def d(self):
        """Turn off dithering - default: False"""
        return self._d

    @d.setter
    def d(self, value):
        self._d.value = value

    @property
    def c(self):
        """This option allows fewer than 256 colors (and fewer than 8 bits per pixel)."""
        return self._c

    @c.setter
    def c(self, value):
        self._c.value = value

    @property
    def g(self):
        """Gamma correction - default: 2.2

        This option specifies the exponent used in gamma correction.
        An exponent of 1.0 turns gamma correction off.
        """
        return self._g

    @g.setter
    def g(self, value):
        self._g.value = value

    @property
    def e(self):
        """Exposure compensation

        This option specifies an exposure compensation in f-stops (powers of two).
        Only integer stops are allowed, for efficiency.
        """
        return self._e

    @e.setter
    def e(self, value):
        if not self._e:
            self._e.value = None
        elif math.log2(value).is_integer():
            self._e.value = value
        else:
            raise ValueError('Only integers that are to power of two are allowed.'
                             ' You provided "{}".'.format(value))

    @property
    def n(self):
        """Sampling factor for large images

        This option specifies a sampling factor for neural network color quantization.
        This value should be between 1 and 80, where 1 takes the longest and produces
        the best results in small areas of the image. If no value is given, a faster
        median cut algorithm is used.
        """
        return self._n

    @n.setter
    def n(self, value):
        self._n.value = value
        
    

