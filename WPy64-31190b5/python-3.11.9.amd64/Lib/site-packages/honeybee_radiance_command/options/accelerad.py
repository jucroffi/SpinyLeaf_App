from .optionbase import OptionCollection, BoolOption, NumericOption,\
    IntegerOption


class AcceleradOptions(OptionCollection):
    """Accelerad command options.

    Also see: https://nljones.github.io/Accelerad/documentation.html#commandline
    """

    __slots__ = ('_g', '_gv', '_al', '_ag', '_az', '_ac', '_an', '_at', '_ax')

    def __init__(self):
        """Accelerad command options.

        Usage:

            options = AcceleradOptions()
            options.ac = 8192
            print(options.to_radiance())
            -ac 8192

            options.g = False
            print(options.to_radiance())
            -ac 8192 -u-
        """
        OptionCollection.__init__(self)
        self._g = BoolOption('g', 'enable or disable GPU ray tracing - default: on')
        self._gv = IntegerOption(
            'gv', 'verbosity of GPU debugging level - default: 0', min_value=0,
            max_value=3)
        self._al = IntegerOption('al', 'spacing between seed point pixels - default: 0')
        self._ag = IntegerOption('ag', 'ambient divisions for final gather infill - default: -1')
        self._az = IntegerOption('az', 'seeds points for ambient samples - default: 0')
        self._ac = IntegerOption('ac', 'k-means clusters for ambient calculation - default: 4096')
        self._an = IntegerOption('an', 'maximum k-means iterations - default: 100')
        self._at = NumericOption('at', 'k-means threshold - default: 0.05')
        self._ax = NumericOption('ax', 'weighting factor in k-means calculation - default: 1.0')

    @property
    def g(self):
        """Enable or disable GPU ray tracing - default: on

        Enable or disable GPU ray tracing. A value of -g- will cause the Accelerad
        programs to revert to normal Radiance behavior without using the GPU.
        """
        return self._g

    @g.setter
    def g(self, value):
        self._g.value = value

    @property
    def gv(self):
        """Flush interval - default: 0

        Set the verbosity of GPU debugging to level. Level 0 produces the fastest
        output but will not display GPU errors which could affect results. Level 1
        reports GPU errors. Levels 2 and 3 provide additional GPU stats but result
        in longer computations.
        """
        return self._gv

    @gv.setter
    def gv(self, value):
        self._gv.value = value

    @property
    def al(self):
        """Spacing between seed point pixels - default: 0

        Set the spacing between seed point pixels for ambient sampling to stride
        in rpict only. A value of zero will cause all pixels to be considered.
        This option is ignored when the -az option is used.
        """
        return self._al

    @al.setter
    def al(self, value):
        self._al.value = value

    @property
    def ag(self):
        """Ambient divisions for final gather infill - default: -1
        
        Set number of ambient divisions for final gather infill to N. When -aa is
        non-zero, N ambient samples will be taken at points not covered by the
        precomputed irradiance cache. A value of -1 will cause the value to be
        copied from -ad."""
        return self._ag

    @ag.setter
    def ag(self, value):
        self._ag.value = value

    @property
    def az(self):
        """Seeds points for ambient samples - default: 0

        Set the number of seeds points for ambient samples to take around the
        circumference of a sphere based at the view point to res in rpict only.
        A value of zero will cause view-dependent seeding to be used instead.
        Thus, zero should not be used in combination with the -S option in which
        a view file changes the view direction from between frames.
        """
        return self._az

    @az.setter
    def az(self, value):
        self._az.value = value

    @property
    def ac(self):
        """k-means clusters for ambient calculation - default: 4096

        Set the number of k-means clusters for ambient calculation to N.
        """
        return self._ac

    @ac.setter
    def ac(self, value):
        self._ac.value = value

    @property
    def an(self):
        """maximum k-means iterations - default: 100

        Set the maximum number of k-means iterations to N. Larger values can cause
        k-means calculation to take longer but will generate more accurate ambient
        results.
        """
        return self._an

    @an.setter
    def an(self, value):
        self._an.value = value

    @property
    def at(self):
        """k-means threshold - default: 0.05

        Set the k-means threshold to thresh. This is the fraction of seeds that
        must change cluster in order for k-means iteration to continue. Smaller
        values can cause k-means calculation to take longer but will generate
        more accurate ambient results.
        """
        return self._at

    @at.setter
    def at(self, value):
        self._at.value = value

    @property
    def ax(self):
        """Weighting factor in k-means calculation - default: 1.0

         	Set the weighting factor for position in k-means error calculation to
            wt. Small values concentrate more ambient calculations around edges
            where ambient gradients are likely to be large.
        """
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax.value = value
