"""psign command."""

from .options.psign import PsignOptions
from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing


class Psign(Command):
    """Psign command.

    Psign produces a RADIANCE picture of the given text. The output dimensions
    are determined by the character height, aspect ratio, number of lines and
    line length. (Also the character size if text squeezing is used.)

    Args:
        options: Command options. It will be set to Radiance default values
            if unspecified.
        output: File path to the output file (Default: None).
        text: Text which will be converted into an image (Default: None).

    Properties:
        * options
        * output
        * text
    """

    __slots__ = ('_text',)

    def __init__(self, options=None, output=None, text=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.options = options
        self._text = text

    @property
    def options(self):
        """psign options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = PsignOptions()

        if not isinstance(value, PsignOptions):
            raise ValueError('Expected Psign options not {}'.format(value))

        self._options = value

    @property
    def text(self):
        """Text string to be converted into an image."""
        return self._text

    @text.setter
    def text(self, value):
        self._text = str(value)

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

        if not stdin_input and self.text:
            cmd = '%s "%s"' % (cmd, self.text)

        if self.pipe_to:
            cmd = '%s | %s' % (cmd, self.pipe_to.to_radiance(stdin_input=True))

        elif self.output:
            cmd = '%s > %s' % (cmd, self.output)

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)
        if not stdin_input and not self.text:
            raise exceptions.MissingArgumentError(self.command, 'text')
