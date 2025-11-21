# coding: utf-8

"""ra_gif command."""

from .options.ra_gif import Ra_GIFOptions
from ._command import Command
import honeybee_radiance_command._typing as typing
import honeybee_radiance_command._exception as exceptions


class Ra_GIF(Command):
    """Ra_gif command.

    Ra_gif converts from RADIANCE to Compuserve GIF color-mapped, compressed
    image files. In the default mode, a RADIANCE picture is converted to a
    color-mapped GIF file of the same horizontal and vertical dimensions with
    8-bits per pixel.

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

    __slots__ = ('_input')

    def __init__(self, options=None, output=None, input=None):
        Command.__init__(self, output=output)
        self._input = input
        self.options = options

    @property
    def options(self):
        """Ra_GIF options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = Ra_GIFOptions()

        if not isinstance(value, Ra_GIFOptions):
            raise ValueError('Expected RaGIF options not {}'.format(value))

        self._options = value

    @property
    def input(self):
        """Radiance HDR image."""
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
            stdin_input: A boolean that indicates if the input for this command comes
                from stdin. This is for instance the case when you pipe the input
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
            cmd = '%s > %s' % (cmd, self.output)

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)
        if not stdin_input and not self.input:
            raise exceptions.MissingArgumentError(self.command, 'input')
            
    
