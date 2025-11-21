"""pfilt command."""

from .options.pfilt import PfiltOptions
from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing


class Pfilt(Command):
    """Pfilt command.

    Pfilt performs anti-aliasing and scaling on a RADIANCE picture. The program
    makes two passes on the picture file in order to set the exposure to the correct
    average value.

    Args:
        options: Command options. It will be set to Radiance default values
            if unspecified.
        output: File path to the output file (Default: None).
        input: File path to the radiance generated hdr file (Default: None).

    Properties:
        * options
        * output
        * input
    """

    __slots__ = ('_input',)

    def __init__(self, options=None, output=None, input=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.options = options
        self._input = input

    @property
    def options(self):
        """pfilt options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = PfiltOptions()

        if not isinstance(value, PfiltOptions):
            raise ValueError('Expected Pfilt options not {}'.format(value))

        self._options = value

    @property
    def input(self):
        """Radiance HDR image file."""
        return self._input

    @input.setter
    def input(self, value):
        if value[-4:].lower() not in ('.hdr', '.pic', '.unf'):
            raise ValueError('"{}" does not have the expected extension for a Radiance '
                             'generated HDR.'.format(type(value)))
        else:
            self._input = typing.normpath(value)

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command. (Default: False).
        """
        self.validate(stdin_input)

        command_parts = [self.command, self.options.to_radiance()]
        cmd = ' '.join(command_parts)

        if not stdin_input and self.input:
            cmd = '%s %s' % (cmd, self.input)

        if self.pipe_to:
            cmd = '%s | %s' % (cmd, self.pipe_to.to_radiance(stdin_input=True))

        elif self.output:
            cmd = '%s > %s' % (cmd, self.output)

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)
        if not stdin_input and not self.input:
            raise exceptions.MissingArgumentError(self.command, 'input')
