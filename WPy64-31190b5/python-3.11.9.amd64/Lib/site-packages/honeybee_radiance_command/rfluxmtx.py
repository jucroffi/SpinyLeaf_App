"""rfluxmtx command"""

from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing
from .options.rfluxmtx import RfluxmtxOptions, RfluxmtxControlParameters


class Rfluxmtx(Command):
    """
    Rfluxmtx command.

    Rfluxmtx samples rays uniformly over the surface given in sender.rad and
    records rays arriving at surfaces in the file receivers.rad, producing a
    flux transfer matrix per receiver. A system octree to which the receivers
    will be appended may be given with a -i option following the receiver file.
    Additional system surfaces may be given in one or more system.rad files,
    which are compiled before the receiver file into an octree sent to the
    rcontrib program to do the actual work.

    Args:
    options: Command options. It will be set to Radiance default values
        if unspecified.
    output: Output file (Default: None).
    octree: Octree file (Default: None).
    sensors: Sensors file (Default: None).
    receivers: Receivers file (Default: None).
    sender: Sender file (Default: None).
    system: System file (Default: None).


    Properties:
        * options
        * output
        * octree
        * sensors
        * system
        * sender
        * receivers

    Note:
    https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/rfluxmtx.pdf

    """

    __slots__ = ('_input', '_sensors', '_sender', '_octree', '_receivers', '_system')

    def __init__(self, options=None, output=None, sensors=None, sender=None,
                 receivers=None, system=None, octree=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.options = options
        self.sensors = sensors
        self.octree = octree
        self.sender = sender
        self.receivers = receivers
        self.system = system

    @property
    def options(self):
        """Rfluxmtx options."""
        return self._options

    @options.setter
    def options(self, value):
        if not value:
            value = RfluxmtxOptions()

        if not isinstance(value, RfluxmtxOptions):
            raise ValueError('Expected Rfluxmtx options not {}'.format(value))

        self._options = value

    @property
    def sensors(self):
        """Sensor file."""
        return self._sensors

    @sensors.setter
    def sensors(self, value):
        if value is None:
            self._sensors = None
        else:
            self._sensors = typing.normpath(value)

    @property
    def sender(self):
        """Sender file."""
        return self._sender

    @sender.setter
    def sender(self, value):
        if value is None:
            self._sender = None
        else:
            self._sender = typing.normpath(value)

    @property
    def receivers(self):
        """Receivers file."""
        return self._receivers

    @receivers.setter
    def receivers(self, value):
        if value is None:
            self._receivers = None
        else:
            self._receivers = typing.normpath(value)

    @property
    def system(self):
        """System file.

        Note that rfluxmtx can accept any number of system files, however, to
        keep the implementation clean, only one system file is being
        allowed."""
        return self._system

    @system.setter
    def system(self, value):
        if value is None:
            self._system = None
        else:
            self._system = typing.normpath(value)

    @property
    def octree(self):
        """Octree file."""
        return self._octree

    @octree.setter
    def octree(self, value):
        if value is None:
            self._octree = None
        else:
            self._octree = typing.normpath(value)

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command (default: False).
        """
        self.validate(stdin_input)

        command_parts = [self.command, self.options.to_radiance()]
        command_parts += [self.sender or "-"]
        command_parts += [self.receivers]
        command_parts += ["-i", '"""%s"""' % self.octree] if self.octree else []
        command_parts += [self.system] if self.system else []

        cmd = ' '.join(command_parts)

        if not stdin_input and self.sensors:
            cmd = ' < '.join((cmd, self.sensors))
        if self.pipe_to:
            cmd = ' | '.join((cmd, self.pipe_to.to_radiance(stdin_input=True)))
        elif self.output:
            cmd = ' > '.join((cmd, self.output))

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)
        if self.receivers is None:
            raise exceptions.MissingArgumentError(self.command, 'receivers')
        if not stdin_input and not self.sensors and not self.sender:
            raise exceptions.MissingArgumentError(self.command, 'sensors')
