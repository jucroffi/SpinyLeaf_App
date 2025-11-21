# coding: utf-8

from .optionbase import (
    OptionCollection,
    NumericOption,
    StringOptionJoined,
    TupleOption,
    FileOption
)
import warnings

class DcglareOptions(OptionCollection):
    """dcglare options.

    Also see: https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/dcglare.pdf/view
    """

    __slots__ = (
        "_l",
        "_b",
        "_vf",
        "_vd",
        "_vu",
        "_vi",
        "_sf",
        "_ss",
        "_se",
    )

    def __init__(self):
        """dcglare command options."""

        OptionCollection.__init__(self)
        self._l = NumericOption("l", "Limit for glare occurrence")
        self._b = NumericOption("b", "Threshold factor in cd/m2 - default: 2000")
        self._vf = FileOption("vf", "View file for DGP calculation")
        self._vd = TupleOption(
            "vd", "View forward vector", None, 3, float
        )
        self._vu = TupleOption(
            "vu", "View up vector - default: 0.000000 0.000000 1.000000", None, 3, float
        )
        self._vi = StringOptionJoined(
            "vi",
            "Format of view file - default: via",
            valid_values=["a", "f", "d"],
            whole=False)
        self._sf = FileOption("sf", "Occupancy schedule file")
        self._ss = NumericOption("ss", "Occupancy start hour")
        self._se = NumericOption("se", "Occupancy end hour")
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.
        """
        if self._vf.is_set and self._vu.is_set:
            warnings.warn(
                'Both -vf and -vu are set. %s will ignore -vu.' % self.command)
        if self._ss.is_set and not self._l.is_set:
            warnings.warn(
                '-ss is set but -l is not. %s will ignore -ss if -ls is not set.'
                % self.command)
        if self._se.is_set and not self._l.is_set:
            warnings.warn(
                '-se is set but -l is not. %s will ignore -se if -ls is not set.'
                % self.command)

    @property
    def l(self):
        """Limit for glare occurrence

        Set the limit for glare occurrence. When this option is provided, the program
        calculates glare autonomy, where any DGP value at or above the limit indicates
        the presence of glare. If the option is not provided, the program calculates DGP
        under each sky condition in the sky matrix instead.
        """
        return self._l

    @l.setter
    def l(self, value):
        self._l.value = value

    @property
    def b(self):
        """Threshold factor in cd/m2 - default: 2000

        If factor is larger than 100, it is used as constant threshold in cd/m2. If 
        factor is less or equal than 100, this factor multiplied by the average luminance 
        in each view will be used as threshold for detecting the glare sources (not 
        recommended). The default value is 2000 (fixed threshold method).
        """
        return self._b

    @b.setter
    def b(self, value):
        self._b.value = value

    @property
    def vf(self):
        """View file for DGP calculation

        Get the list of views for DGP calculation from file. Each line in file contains 
        six numeric values corresponding to the position and direction of a view. 
        Generally, this is the same file that is used as input to rcontrib to create the 
        daylight coefficient matrices.
        """
        return self._vf

    @vf.setter
    def vf(self, value):
        self._vf.value = value

    @property
    def vd(self):
        """View forward vector for DGP calculation

        Set the view forward vector (vertical direction) for DGP calculation to xd yd zd.
        This option is ignored when the −vf option is provided.
        """
        return self._vd

    @vd.setter
    def vd(self, value):
        self._vd.value = value

    @property
    def vu(self):
        """View up vector for DGP calculation - default: 0.000000 0.000000 1.000000

        Set the view up vector (vertical direction) for DGP calculation to xd yd zd. The 
        default up vector is the positive z direction.
        """
        return self._vu

    @vu.setter
    def vu(self, value):
        self._vu.value = value

    @property
    def vi(self):
        """Format of view file - default: 'a'
        
        Set the format of the view file. Available options are 'f' for single and 'd'
        for double precison IEEE float. The default when no value is provided is to use 
        ASCII.
        """
        return self._vi

    @vi.setter
    def vi(self, value):
        self._vi.value = value

    @property
    def sf(self):
        """Occupancy schedule file
        
        Set the occupancy schedule file. In the event that the sky matrix includes 
        unoccupied hours that should not contribute to the glare autonomy calculation, 
        occupancy schedule file will be read to determine which entries from the sky file
        matrix will be included in this calculation. Each line of occupancy schedule file
        is expected to contain a numeric value at the end of a comma-delimited list, with
        zero corresponding to unoccupied. This argument is used only if -l is specified.
        """
        return self._sf

    @sf.setter
    def sf(self, value):
        self._sf.value = value

    @property
    def ss(self):
        """Occupancy start hour
        
        Set the occupancy start hour. This option is provided for expediency when no 
        occupancy schedule file is available. It is assumed that the sky matrix includes
        24 entries per day, corresponding to one per hour. This argument is used only if
        -l is specified
        """
        return self._ss

    @ss.setter
    def ss(self, value):
        self._ss.value = value

    @property
    def se(self):
        """Occupancy end hour
        
        Set the occupancy end hour. This option is provided for expediency when no
        occupancy schedule file is available. It is assumed that the sky matrix includes
        24 entries per day, corresponding to one per hour. This argument is used only if
        -l is specified.
        """
        return self._se

    @se.setter
    def se(self, value):
        self._se.value = value
