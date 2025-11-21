"""rcollate command."""

from .options.rcollate import RcollateOptions
from ._command import Command

import honeybee_radiance_command._exception as exceptions
from ._typing import path_checker


class Rcollate(Command):
    """Rcollate command.

    Rcollate reads in a single matrix file (table) and reshapes it to have the
    number of columns specified by the -oc option.

    Args:
        options: Command options. It will be set to Radiance default values if not
            provided by user.
        output: File path to the output matrix file (Default: None).
        input: File path of the input matrix file. (Default: None).

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
        """rcollate options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = RcollateOptions()

        if not isinstance(value, RcollateOptions):
            raise ValueError('Expected RcollateOptions not {}'.format(type(value)))

        self._options = value

    @property
    def input(self):
        """Path to the matrix file on which the reshaping and other operations are to
        be performed."""
        return self._input

    @input.setter
    def input(self, value):
        self._input = path_checker(value)

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
