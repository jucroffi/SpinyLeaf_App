"""Base class for Radiance commands.

A Radiance command can be simplified as

    `command [ options ] [ $env variable ] [ @file ] additional-inputs `.

Output are always printed to stdout and can be redirected to a file or piped to another
command. Honeybee implementation follows Radiance command structure closely.

Baseclass arguments are `options` and `output`. Each command will accept additional input
arguments. These input arguments can be one or several depending on the command. If
options are not provided by user default options for command will be used. If output is
not provided the results will be sent to standard output.

Commands also support piping and you can pipe the results from one command to another.
Honeybee takes care of adjusting the command semantics based on Radiance standards.

Honeybee commands don't support environmental variable as part of the command itself. You
can pass the environmental variables that are needed to run the command during the run.
For more information see `run` method documentation.

Except for Radiance Options commands are supposed to work with files and folders and not
Honeybee objects. This means that you need to convert your model to Radiance files before
using the commands. See the implementation of daylight recipes for comprehensive
examples.

Example:

```
# initiate Radiance command
rtrace = Rtrace()

# update command options
rtrace.options.ab = 3
rtrace.options.I = True

# alternatively you can update the options from a string
# rtrace.options.update_from_string('-ab 3 -I')

# add octree
rtrace.octree = 'scene.oct'

# add sensor grid
rtrace.input = 'sensors.pts'

# add rcalc command to post process the results
rcalc = Rcalc()
rcalc.e = '$1=(0.265*$1+0.67*$2+0.065*$3)*179'
rcalc.output = '5-output/illuminance.dat'

# pipe results from rtrace to rcalc
rtrace.pipe_to(rcalc)

# here is how the command looks like
print(rtrace.to_radiance())

#

# run the command. You can pass environment variables to command here
rtrace.run(env)
```

"""

import warnings
import os

from .options import OptionCollection
from ._command_util import run_command
import honeybee_radiance_command._typing as typing


class Command(object):
    """Base class for Radiance commands."""

    __slots__ = ('_options', '_output', '_pipe_to_command')

    def __init__(self, options=None, output=None):
        """Radiance command.

        Args:
            options: Command options.
            output: Path to output file.
        """
        # this should be overwritten in subclasses but I'm adding a minimum check here in
        # case someone forgets to add it to one of the subclasses
        if options:
            assert isinstance(options, OptionCollection), \
                'Options must be an instance of OptionCollection class.'
        self._options = options
        self.output = output
        self._pipe_to_command = None

    @property
    def command(self):
        return self.__class__.__name__.lower()

    @property
    def options(self):
        """command options."""
        return self._options

    @property
    def output(self):
        """output file."""
        return self._output

    @output.setter
    def output(self, value):
        if value:
            value = typing.normpath(value)
        self._output = value

    @property
    def pipe_to(self):
        """Second command to pipe the outputs from this command."""
        return self._pipe_to_command

    @pipe_to.setter
    def pipe_to(self, command):
        if command is not None and not isinstance(command, Command):
            ValueError(
                'The output can only be piped to another Command not {}'.format(
                    type(command)
                )
            )
        self._pipe_to_command = command
        self.validate()
    
    def enclose_command(self, stdin_input=False):
        """Enclose command in quotes and exclamation point ('!'). This method should be 
        used when reading the input of a command from another Radiance command.
        
        Example:
        rmtxop -c 47.4 119.9 11.6 "!rmtxop view transmission daylight sky" > output
        """
        if os.name == 'posix':
            return '\'!%s\'' % self.to_radiance(stdin_input)
        else:
            return '\"!%s\"' % self.to_radiance(stdin_input)

    def to_radiance(self, stdin_input=False):
        """Radiance command.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command (default: False).
        """
        command_parts = [self.command]
        if self.options:
            command_parts.append(self.options.to_radiance())
        command = ' '.join(command_parts).replace('\\', '/')
        if self.pipe_to:
            return ' | '.join(
                (command, self.pipe_to.to_radiance(stdin_input=True))
            ).replace('\\', '/')
        elif self.output:
            return ' > '.join((command, self.output)).replace('\\', '/')
        else:
            return command.replace('\\', '/')

    def validate(self):
        """Overwrite this method to add extra specific checks for the command.
        For instance for rcontrib you want to make sure there is at least one
        modifier set in the command.

        This method will be executed right before running the command.
        """
        if self.output and self.pipe_to:
            warnings.warn(
                '%s: "output" and "pipe_to" are both set for this command.'
                ' The "output" argument  will be ignored.'
                % self.__class__.__name__.lower()
            )

    def run(self, env=None, cwd=None):
        """Run command as a subprocess.

        Args:
            env: Environmental variables (default: None).
            cwd: Working directory (Default: '.').

        Returns:
            - int: Command return code.
        """
        cmd = self.to_radiance().replace('\\', '/')
        rc = run_command(cmd, env, cwd)
        self.after_run()
        return rc

    def after_run(self):
        """After run script.

        Overwrite this method to add extra tasks that runs right after run method.
        """
        pass

    def __repr__(self):
        return self.to_radiance()
