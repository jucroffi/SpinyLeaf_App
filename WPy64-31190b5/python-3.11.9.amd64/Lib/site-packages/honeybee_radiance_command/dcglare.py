"""dcglare command."""

from .options.dcglare import DcglareOptions
from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing


class Dcglare(Command):
    """Dcglare command.

    Dcglare generates daylight glare probability (DGP) predictions for multiple points in
    a space under a variety of daylit conditions. Usually, it is used to produce hourly
    DGP values for an entire year, or if the -l option is provided, it calculates glare
    autonomy based on an annual occupancy schedule.

    As input, dcglare requires daylight coefficient matrices relating the illuminance at
    each view point to the brightness of each sky patch. Two such matrices are required.
    The first, DCdirect, consists of direct views to the sky only and is calculated by
    rcontrib using a single ambient bounce. The second, DCtotal, includes the total
    direct and diffuse contribution of each sky patch. The latter can be calculated
    directly by rcontrib as in the two-phase method, or internally as in the three-phase
    method if given view, BSDF, and daylight matrices. In this respect, dcglare is
    similar to dctimestep except that it calculates DGP instead of irradiance. The final
    input is the sky contribution matrix, usually computed by gendaymtx, which may be
    passed on the standard input. For efficiency, matrices stored in files can be
    represented as binary float data if machine byte-order is not an issue.

    Args:
        options: Command options. It will be set to Radiance default values
            if unspecified.
        output: File path to the output file (Default: None).
        dc_direct: File path to the direct contribution file (Default: None).
        dc_total: File path to the total (direct and diffuse) contribution file
            (Default: None).
        sky_matrix: File path to the sky contribution file (Default: None).
        vmtx: File path to view matrix (three-phase methoc) (Default: None).
        dmtx: File path to daylight matrix (three-phase methoc) (Default: None).
        tmtx: File path to transmission matrix (three-phase methoc) (Default: None).

    Properties:
        * options
        * output
        * dc_direct
        * dc_total
        * sky_matrix
        * vmtx
        * dmtx
        * tmtx
    """

    __slots__ = ('_dc_direct', '_dc_total', '_sky_matrix', '_vmtx', '_dmtx', '_tmtx')

    def __init__(self, options=None, output=None, dc_direct=None, dc_total=None,
                 sky_matrix=None, vmtx=None, dmtx=None, tmtx=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.options = options
        self.dc_direct = dc_direct
        self.dc_total = dc_total
        self.sky_matrix = sky_matrix
        self.vmtx = vmtx
        self.dmtx = dmtx
        self.tmtx = tmtx

    @property
    def options(self):
        """dcglare options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = DcglareOptions()

        if not isinstance(value, DcglareOptions):
            raise ValueError('Expected Dcglare options not {}'.format(value))

        self._options = value

    @property
    def dc_direct(self):
        """Direct contribution matrix."""
        return self._dc_direct

    @dc_direct.setter
    def dc_direct(self, value):
        if value is None:
            self._dc_direct = None
        else:
            self._dc_direct = typing.normpath(value)

    @property
    def dc_total(self):
        """Total (direct and diffuse) contribution matrix."""
        return self._dc_total

    @dc_total.setter
    def dc_total(self, value):
        if value is None:
            self._dc_total = None
        else:
            self._dc_total = typing.normpath(value)

    @property
    def sky_matrix(self):
        """Sky contribution matrix."""
        return self._sky_matrix

    @sky_matrix.setter
    def sky_matrix(self, value):
        if value is None:
            self._sky_matrix = None
        else:
            self._sky_matrix = typing.normpath(value)

    @property
    def vmtx(self):
        """View matrix."""
        return self._vmtx

    @vmtx.setter
    def vmtx(self, value):
        if value is None:
            self._vmtx = None
        else:
            self._vmtx = typing.normpath(value)

    @property
    def dmtx(self):
        """Daylight matrix."""
        return self._dmtx

    @dmtx.setter
    def dmtx(self, value):
        if value is None:
            self._dmtx = None
        else:
            self._dmtx = typing.normpath(value)

    @property
    def tmtx(self):
        """Transmission matrix (BSDF)."""
        return self._tmtx

    @tmtx.setter
    def tmtx(self, value):
        if value is None:
            self._tmtx = None
        else:
            self._tmtx = typing.normpath(value)

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command. (Default: False).
        """
        self.validate(stdin_input)

        command_parts = [self.command, self.options.to_radiance()]
        command_parts += [self.dc_direct]
        command_parts += [self.dc_total] if not self.tmtx else \
            [self.vmtx, self.tmtx, self.dmtx]
        if not stdin_input:
            command_parts += [self.sky_matrix]
        cmd = ' '.join(command_parts)

        if self.pipe_to:
            cmd = '%s | %s' % (cmd, self.pipe_to.to_radiance(stdin_input=True))

        elif self.output:
            cmd = '%s > %s' % (cmd, self.output)

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)
        if not self.dc_direct:
            raise exceptions.MissingArgumentError(self.command, 'dc_direct')
        if not self.dc_total and not self.tmtx:
            raise exceptions.MissingArgumentError(self.command, 'dc_total')
        if self.tmtx and not self.vmtx:
            raise exceptions.MissingArgumentError(self.command, 'vmtx')
        if self.tmtx and not self.dmtx:
            raise exceptions.MissingArgumentError(self.command, 'dmtx')
        if not stdin_input and not self.sky_matrix:
            raise exceptions.MissingArgumentError(self.command, 'sky_matrix')
