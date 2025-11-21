# coding: utf-8

"""Gensky command."""

from .options.gensky import GenskyOptions
from ._command import Command
from ._typing import tuple_with_length, int_in_range
import honeybee_radiance_command._exception as exceptions


class Gensky(Command):
    """Gensky Command.

    Gensky produces a RADIANCE scene description for the CIE standard sky
    distribution at the given month, day and time. By default, the time is
    interpreted as local standard time on a 24-hour clock. The time value may
    be given either as decimal hours, or using a colon to separate hours and minutes.

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

    Properties
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

        Command.__init__(self, output=output)
        self.options = options
        self.month = month
        self.day = day
        self.time = time
        self.time_zone = time_zone
        self.solar_time = solar_time

    @property
    def options(self):
        """Gensky options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = GenskyOptions()

        if not isinstance(value, GenskyOptions):
            raise ValueError('Expected GenskyOptions not {}'.format(type(value)))

        self._options = value

    @property
    def month(self):
        """Month."""
        return self._month

    @month.setter
    def month(self, value):
        if not value:
            self._month = None
        else:
            self._month = int_in_range(value, 1, 12)

    @property
    def day(self):
        """Day."""
        return self._day

    @day.setter
    def day(self, value):
        if not value:
            self._day = None
        else:
            self._day = int_in_range(value, 1, 31)

    @property
    def time(self):
        """time."""
        return self._time

    @time.setter
    def time(self, value):
        if not value:
            self._time = None

        elif isinstance(value, float):
            hour, minute = str(value).split('.')

            # Validate hour
            hour = int_in_range(int(hour), 0, 23)

            # Validate minute
            if minute in ('25', '5', '75'):
                minutes = {'25': '15', '5': '30', '75': '45'}

                self._time = '%s:%s' % (hour, minutes[minute])
            else:
                raise ValueError(
                    'Decimal values are only allowed for 15 minute increments.'
                    ' Such as 9.25, 9.5, and 9.75 which will become 9:15, 9:30 and'
                    ' 9:45 respectively. You provided %s' % (value)
                )

        elif isinstance(value, str) and ':' in value:
            hour, minute = value.split(':')

            # Validate hour
            hour = int_in_range(int(hour), 0, 23)
            # Validate minute
            minute = int_in_range(int(minute), 0, 59)

            self._time = value

        else:
            raise ValueError(
                '%s is not a valid format. Examples of acceptable formats are'
                ' a float value of 21.5 or a string value of 21:30' % (value)
            )


    @property
    def time_zone(self):
        """time zone."""
        return self._time_zone

    @time_zone.setter
    def time_zone(self, value):
        time_zones = [
            'YST', 'PST', 'MST', 'CST', 'EST', 'GMT', 'CET', 'EET', 'AST', 'GST', 'IST',
            'JST', 'NZST', 'YDT', 'PDT', 'MDT', 'CDT', 'EDT', 'BST', 'CEST', 'EEST',
            'ADT', 'GDT', 'IDT', 'JDT', 'NZDT']
        if not value:
            self._time_zone = None
        elif value.upper() in time_zones:
            self._time_zone = value
        else:
            raise ValueError(
                'Time zone must a three letter string from the following'
                ' options %s.' % (time_zones)
            )

    @property
    def solar_time(self):
        """Whether solar time is requested."""
        return self._solar_time

    @solar_time.setter
    def solar_time(self, value):
        if not value:
            self._solar_time = None
        elif isinstance(value, bool):
            self._solar_time = value
        else:
            raise ValueError('Solar time only accepts True or False as a value.')

    @property
    def input(self):
        """Input string for gensky."""

        if not self.month and not self.day and not self.time:
            self._input = None
            return self._input

        # Only month, day and time provided as arguments
        elif not (self.time_zone and self.solar_time):
            self._input = '%s %s %s' % (str(self.month), str(self.day), str(self.time))
            return self._input

        # Time zone is provided
        elif self.time_zone and not self.solar_time:
            self._input = '%s %s %s%s' % (str(self.month), str(self.day),
                                          str(self.time), self.time_zone)
            return self._input

        # Local solar time requested
        elif self.solar_time and not self.time_zone:
            self._input = '+%s %s %s' % (str(self.month), str(self.day),
                                         str(self.time))
            return self._input

        # Time zone and solar time both requested
        elif self.time_zone and self.solar_time:
            self._input = '+%s %s %s%s' % (str(self.month), str(self.day),
                                           str(self.time), self.time_zone)
            return self._input

    @classmethod
    def from_ang(cls, angles, options=None):
        """Create a Gensky command using sun altitude and azimuth angles in degrees.

        The altitude is measured in degrees above the horizon, and the azimuth is
        measured in degrees west of South.

        Args:
            angles: A tuple of altitude and azimuth angles.
            options: Command options. It will be set to Radiance default values if not
                provided by user.
        """
        cls_from_ang = cls()

        if not options:
            options = cls_from_ang.options

        options.ang = tuple_with_length(angles, length=2)

        return cls_from_ang

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command (default: False).
        """
        # If form_ang is not used, validate arguments
        if not self.options.ang.is_set:
            self.validate()

        # Month, day, and time are set and then -ang option is set
        elif self.options.ang.is_set and (self.month and self.day and self.time):
            raise ValueError(
                'Gensky command can be used with either month, day, time or with'
                ' -ang option that uses sun altitude, azimuth. Setting both are'
                ' not allowed.')

        command_parts = [self.command]

        if self.options:
            command_parts.append(self.options.to_radiance())

        cmd = ' '.join(command_parts)

        # This will happen when from_ang method is used
        if not stdin_input and not self.input:
            cmd = '%s' % (cmd)

        if not stdin_input and self.input:
            cmd = '%s %s' % (cmd, self.input)

        if self.output:
            cmd = '%s > %s' % (cmd, self.output)

        return ' '.join(cmd.split())

    def validate(self):
        Command.validate(self)
        if not self.month:
            raise exceptions.MissingArgumentError(self.command, 'month')
        if not self.day:
            raise exceptions.MissingArgumentError(self.command, 'day')
        if not self.time:
            raise exceptions.MissingArgumentError(self.command, 'time')

