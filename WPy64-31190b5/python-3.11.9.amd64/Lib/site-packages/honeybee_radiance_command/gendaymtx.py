"""gendaymtx command."""
from .options.gendaymtx import GendaymtxOptions
from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing


class Gendaymtx(Command):
    """Gendaymtx command.

    Gendaymtx takes a weather tape as input and produces a matrix of sky patch values
    using the Perez all weather model. The weather tape is assumed to be in the simple
    ASCII format understood by DAYSIM, which contains a short header with the site
    parameters followed by the month, day, standard time, direct normal and diffuse
    horizontal irradiance values, one time step per line. Each time step line is used
    to compute a column in the output matrix, where rows correspond to sky patch
    positions, starting with 0 for the ground and continuing to 145 for the zenith using
    the default -m 1 parameter setting.

    Args:
        options: Gendaymtx options. It will be set to Radiance default values
            if unspecified.
        output: Path to output file.
        wea: Path to input wea file.

    Properties:
        * options
        * output
        * wea
    """

    __slots__ = ('_wea',)

    def __init__(self, options=None, output=None, wea=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.wea = wea
        self.options = options

    @property
    def options(self):
        """Rtrace options."""
        return self._options

    @options.setter
    def options(self, value):
        if value is None:
            value = GendaymtxOptions()

        if not isinstance(value, GendaymtxOptions):
            raise ValueError('Expected GendaymtxOptions not {}'.format(type(value)))

        self._options = value

    @property
    def wea(self):
        """Wea file."""
        return self._wea

    @wea.setter
    def wea(self, value):
        if value is None:
            self._wea = value
        else:
            self._wea = typing.normpath(value)

    def to_radiance(self):
        """Command in Radiance format."""
        self.validate()

        command_parts = [self.command, self.options.to_radiance(), self.wea]
        cmd = ' '.join(command_parts)
        if self.pipe_to:
            cmd = ' | '.join((cmd, self.pipe_to.to_radiance(stdin_input=True)))
        elif self.output:
            cmd = ' > '.join((cmd, self.output))

        return ' '.join(cmd.split())

    def validate(self):
        Command.validate(self)
        if self.wea is None:
            raise exceptions.MissingArgumentError(self.command, 'wea')
