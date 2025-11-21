"""pinterp command."""

from .options.pinterp import PinterpOptions
from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing


class Pinterp(Command):
    """Pinterp command.

    Pinterp interpolates or extrapolates a new view from one or more RADIANCE pictures 
    and sends the result to the standard output.

    Args:
        options: Command options. It will be set to Radiance default values
            if unspecified.
        output: File path to the output file (Default: None).
        view: File path to a view file. This is the view to interpolate or extrapolate 
            (Default: None).
        image: Radiance HDR image(s) to interpolate or extrapolate from. A list of images
            can be given if multiple images are used (Default: None).
        zspec: The distance from the view point to each pixel in the image(s). Typically
            this input is generated as a file by using the -z option of rpict. A number
            can also be given instead, which should only be used if the view point remains
            constanst, e.g., if the view point in a single input image is equal to that 
            of the input view. If a list of images is given, then the zspec must also be
            a list matching the length of the list of images. (Default: None).

    Properties:
        * options
        * output
        * view
        * image
        * zspec
    """

    __slots__ = ('_view', '_image', '_zspec')

    def __init__(
        self, options=None, output=None, view=None, image=None, zspec=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.options = options
        self.view = view
        self.image = image
        self.zspec = zspec

    @property
    def options(self):
        """pinterp options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = PinterpOptions()

        if not isinstance(value, PinterpOptions):
            raise ValueError('Expected Pinterp options not {}'.format(value))

        self._options = value

    @property
    def view(self):
        """View to interpolate or extrapolate."""
        return self._view

    @view.setter
    def view(self, value):
        if not value:
            self._view = None
        else:
            self._view = typing.normpath(value)

    @property
    def image(self):
        """Radiance HDR image file(s)."""
        return self._image

    @image.setter
    def image(self, value):
        if not isinstance(value, (list, tuple)):
            self._image = [value]
        else:
            self._image = value

    @property
    def zspec(self):
        """z specification for input image(s)"""
        return self._zspec

    @zspec.setter
    def zspec(self, value):
        if not isinstance(value, (list, tuple)):
            self._zspec = [value]
        else:
            self._zspec = value

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command. (Default: False).
        """
        self.validate(stdin_input)

        # length of images and zpec must be the same
        if len(self.image) != len(self.zspec):
            raise ValueError(
                'Pinterp command needs each input image to be accompanied by a'
                ' z specification. Found {} image(s) and {} z specification(s).'
                ' Make sure the number of images are equal to the number of'
                ' z specifications'.format(len(self.image), len(self.zspec)))

        if stdin_input: 
            self.options.vf = '-'
        else: 
            self.options.vf = self.view

        command_parts = [self.command, self.options.to_radiance()]
        cmd = ' '.join(command_parts)

        for (img, z) in zip(self.image, self.zspec):
            cmd = '%s %s %s' % (cmd, img, z)

        if self.pipe_to:
            cmd = '%s | %s' % (cmd, self.pipe_to.to_radiance(stdin_input=True))

        elif self.output:
            cmd = '%s > %s' % (cmd, self.output)

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)
        if not stdin_input and not self.view:
            raise exceptions.MissingArgumentError(self.command, 'view')
        if not self.image[0]:
            raise exceptions.MissingArgumentError(self.command, 'image')
        if not self.zspec[0]:
            raise exceptions.MissingArgumentError(self.command, 'zspec')
