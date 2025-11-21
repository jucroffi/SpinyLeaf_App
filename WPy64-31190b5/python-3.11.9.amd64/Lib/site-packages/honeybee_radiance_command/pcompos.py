"""pcompos command."""

from .options.pcompos import PcomposOptions
from ._command import Command

import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing



class Pcompos(Command):
    """Pcompos command.

    Pcompos arranges and composites RADIANCE pictures and sends the result to
    the standard output.

    Args:
        options: Command options. It will be set to Radiance default values if not
            provided by user.
        output: File path to the output file (Default: None).
        input: A list of paths to radiance generated hdr images. (Default: None).

    Properties:
        * options
        * output
        * input
    """

    __slots__ = ('_input')

    def __init__(self, options=None, output=None, input=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.input = input
        self.options = options

    @property
    def options(self):
        """pcompos options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = PcomposOptions()

        if not isinstance(value, PcomposOptions):
            raise ValueError('Expected PcomposOptions not {}'.format(type(value)))

        self._options = value

    @property
    def input(self):
        """A string of joined paths to the hdr files."""
        return self._input

    @input.setter
    def input(self, value):
        if not value:
            value = []
        elif not isinstance(value, (list, tuple)):
            value = [value]
        for image in value:
            if image[-4:].lower() not in ('.hdr', '.pic', '.unf'):
                raise ValueError(
                    'A list of .hdr files required. Instead got %s.' % (value)
                )
        self._input = ' '.join(typing.normpath(path) for path in value)

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command (default: False).
        """
        self.validate(stdin_input)

        command_parts = [self.command, self.options.to_radiance()]
        cmd = ' '.join(command_parts)

        if not stdin_input and self.input:
            cmd = ' '.join((cmd, self.input))

        if self.pipe_to:
            cmd = ' | '.join((cmd, self.pipe_to.to_radiance(stdin_input=True)))

        elif self.output:
            cmd = ' > '.join((cmd, self.output))

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)
        if not stdin_input and not self.input:
            raise exceptions.MissingArgumentError(self.command, 'input')
