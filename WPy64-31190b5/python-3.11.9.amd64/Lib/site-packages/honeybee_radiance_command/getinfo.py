"""getinfo command."""

from .options.getinfo import GetinfoOptions
from ._command import Command

import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing
import warnings


class Getinfo(Command):
    """Getinfo command.

    Getinfo reads the header of each RADIANCE file and writes it to the standard
    output. Octree and picture files are in a binary format, which makes it
    difficult to determine their content. Therefore, a few lines of text are
    placed at the beginning of each file by the RADIANCE program that creates it.
    The end of the header information and the start of the data is indicated
    by an empty line.

    Args:
        options: Command options. It will be set to Radiance default values
            if unspecified.
        output: File path to the output file (Default: None).
        input: A list of paths to radiance generated hdr images, octree files,
            etc. (Default: None).

    Properties:
        * options
        * output
        * input
        * remove_header
    """

    __slots__ = ('_input', '_remove_header')

    def __init__(self, options=None, output=None, input=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self._input = input
        self.options = options
        self._remove_header = False

    @classmethod
    def remove_header(cls, output=None, input=None):
        """Return a class instance explicitly for removing the header. 
        
        This instance returns getinfo with a hyphen that simply removes the header and 
        copies the body of the file from the standard input to the standard output.
        Options a and d will be ignored if this class is used.
        """
        header_cls = cls(output=output, input=input)
        header_cls._remove_header = True
        return header_cls

    @property
    def options(self):
        """getinfo options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = GetinfoOptions()

        if not isinstance(value, GetinfoOptions):
            raise ValueError('Expected GetinfoOptions not {}'.format(type(value)))

        self._options = value

    @property
    def input(self):
        """A string of joined paths to the hdr or octree files."""
        return self._input

    @input.setter
    def input(self, value):
        if not value:
            self._input = []
        elif not isinstance(value, (list, tuple)):
            value = [value]
        self._input = ' '.join(typing.normpath(path) for path in value)

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command (default: False).
        """
        self.validate(stdin_input)

        command_parts = [self.command]
        if not self._remove_header:
            command_parts.append(self.options.to_radiance())
        else:
            command_parts.append('-')

        cmd = ' '.join(command_parts)

        if not stdin_input and self.input:
            if self.options.a or self._remove_header:
                cmd = ' < '.join((cmd, self.input))
            else:
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
        if self._remove_header and self.options.a:
            warnings.warn('getinfo: option a will be ignored when using remove_header')
        if self._remove_header and self.options.d:
            warnings.warn('getinfo: option d will be ignored when using remove_header')
