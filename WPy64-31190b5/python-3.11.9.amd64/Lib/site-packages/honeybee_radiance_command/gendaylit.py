
# coding: utf-8

"""Gendaylit command. This command behaves almost identical to the Gensky command.
Hence, this command inherits from this library's Gensky implementation."""

from .options.gendaylit import GendaylitOptions
from .gensky import Gensky


class Gendaylit(Gensky):
    """Gendaylit Command.

    Gendaylit produces a RADIANCE scene description based on an angular distribution
    of the daylight sources (direct+diffuse) for the given atmospheric conditions
    (direct and diffuse component of the solar radiation), date and local standard
    time. The default output is the radiance of the sun (direct) and the sky (diffuse)
    integrated over the visible spectral range (380-780 nm). We have used the
    calculation of the sun's position and the ground brightness models which
    were programmed in gensky.

    Args:
        month: An integer representing the number of the month. Count starts from 01.
        day: An integer representing the number of the day in a month. Count starts
            from 01.
        time: A string representing hour and minute in 24 hours format.
            Examples of acceptable format are 21.30 and 21:30.
        time_zone: A three letter text representing the time zone.
            Following are acceptable time zones with their corresponding hour
            differences from Greenwhich mean time;
            YST +9, PST +8, MST +7, CST +6, EST +5, GMT 0, CET -1, EET -2, AST -3,
            GST -4, IST -5.5, JST -9, NZST -12, YDT +8, PDT +7, MDT +6, CDT +5,
            EDT +4, BST -1, CEST -2, EEST -3, ADT -4, GDT -5, IDT -6.5, JDT -10,
            NZDT -13
        solar_time: A boolean to use local solar time. If set to True then the time
            is preceded by '+' sign and local solar time is used instead of local
            standard time.
        options: Command options. It will be set to Radiance default values
            if unspecified.
        output: File path to the output file (Default: None).
    
    Properties:
        * options
        * month
        * day
        * time
        * time_zone
        * solar_time
        * input
    """

    __slots__ = ('_month', '_day', '_time', '_time_zone', '_solar_time', '_input')

    def __init__(self, month=None, day=None, time=None, time_zone=None,
                 solar_time=False, options=None, output=None):

        Gensky.__init__(self, month=month, day=day, time=time, time_zone=time_zone,
                        solar_time=solar_time, options=options, output=output)

    @property
    def options(self):
        """Gendaylit options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = GendaylitOptions()

        if not isinstance(value, GendaylitOptions):
            raise ValueError('Expected Gendaylit Options not {}'.format(type(value)))

        self._options = value
