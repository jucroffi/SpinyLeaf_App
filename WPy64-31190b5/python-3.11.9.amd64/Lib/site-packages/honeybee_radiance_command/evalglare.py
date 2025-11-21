"""evalglare command."""

from .options.evalglare import EvalglareOptions
from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing


class Evalglare(Command):
    """Evalglare command.

    Evalglare determines and evaluates glare sources within a 180 degree fisheye image,
    given in the RADIANCE image format (.pic or .hdr).

    The image should be rendered as fisheye (e.g. using the -vta or -vth option)
    using 180 degrees for the horizontal and vertical view angle (-vv 180, -vh 180.)
    Due to runtime reasons of the evalglare code, the image should be smaller
    than 1500x1500 pixels. The recommended size is 1000x1000 pixels, the minimum
    recommended size is 800x800 pixels.

    The program calculates the daylight glare probability (DGP) as well as other
    glare indexes (DGI, DGI_MOD, UGR, UGR_EXP, VCP, CGI, UDP) to the standard
    output. The DGP describes the fraction of persons disturbed caused by glare
    from daylight as a number from 0 to 1, where 0 is no-one disturbed and 1 is
    everyone. Values lower than 0.2 are out of the range of the user assessment
    tests, where the program is based on and should be interpreted carefully.

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
        """evalglare options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = EvalglareOptions()

        if not isinstance(value, EvalglareOptions):
            raise ValueError('Expected Evalglare options not {}'.format(value))

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
