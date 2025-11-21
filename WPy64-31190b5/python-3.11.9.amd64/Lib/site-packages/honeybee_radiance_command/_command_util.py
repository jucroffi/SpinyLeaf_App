"""Utility functions to process and run shell commands."""
from __future__ import print_function  # hello Python 2 - let's say bye soon.

import subprocess
import re
import shlex
import platform
import os
import sys


if sys.version_info[0] < 3:
    STDOUT_CHECK = ''
else:
    STDOUT_CHECK = b''


def run_command(input_command, env=None, cwd=None, mute=True):
    """Run a shell command.
    This function prints both STDOUT and STDERR to the same PIPE. Use shell piping
    to pipe the stdout from the commands to a file.
    Args:
        input_command: Input command.
        env: Additional environmental variable that will be added to global environment.
        cwd: Current working directory. If provided command will be executed from this
            folder.
    """
    if platform.system() == 'Windows':
        command = input_command.replace('\'', '"')
    else:
        command = input_command.replace('"', '\'')

    # change cwd - Popen cwd input simply doesn't work.
    cur_dir = os.getcwd()
    if cwd:
        os.chdir(cwd)

    # update environmental variable
    g_env = os.environ.copy()
    if env:
        for k, v in env.items():
            if k.strip().upper() == 'PATH':
                g_env['PATH'] = os.pathsep.join((v, g_env['PATH']))
            else:
                g_env[k] = v

    g_env['PYTHONUNBUFFERED'] = '1'  # ensure stdout will not be buffered

    command = command.replace('\\', '/')
    if not mute:
        print('running %s' % command)

    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, shell=True, env=g_env
        )
    except Exception as e:
        # this is an edge case that is happening for certain commands on Mac when
        # the command is executed from inside IronPython 2.7
        # see: https://discourse.ladybug.tools/t/hdr-adjust-false-color-not-working-on-mac/16941/3
        if platform.system() == 'Darwin':
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, shell=True, env=g_env
            )
        else:
            raise ValueError(e)

    try:
        for line in iter(process.stdout.readline, STDOUT_CHECK):
            try:
                # Python 3 - almost all the time that we use this library
                print(line.decode('utf-8'), end='')
            except AttributeError:
                # python 2 - line is already a string
                print(line, end='')
    except Exception:  # nothing in stdout
        pass

    _, stderr = process.communicate()
    rc = process.returncode

    if cwd:
        os.chdir(cur_dir)

    try:
        for line in iter(stderr.readline, STDOUT_CHECK):
            try:
                # Python 3 - almost all the time that we use this library
                print(line.decode('utf-8'), end='')
            except AttributeError:
                # python 2 - line is already a string
                print(line, end='')
    except Exception:  # nothing in stderr
        pass

    if rc != 0 and isinstance(rc, int):  # return code not included on Mac
        print(command)
        raise RuntimeError('None zero return code: %d' % rc)

    # only gets here is successful
    return 0


def _process_command(input_command):
    """Process input command before execution.

    This method:
        1. separates output file if the output is redirected to a file using >
        2. breaks-down the command into several commands if piping is happening |
    """
    def _raw(text):
        """Raw string representation.

        This is useful to handle edge cases in Windows.
        """

        escape_dict = {
            '\a': r'\a', '\b': r'\b', '\c': r'\c', '\f': r'\f', '\n': r'\n',
            '\r': r'\r', '\t': r'\t', '\v': r'\v', '\'': r'\'', '\"': r'\"',
            '\0': r'\0', '\1': r'\1', '\2': r'\2', '\3': r'\3', '\4': r'\4',
            '\5': r'\5', '\6': r'\6', '\7': r'\7', '\8': r'\8', '\9': r'\9'
        }

        new_string = ''
        for char in text:
            try:
                new_string += escape_dict[char]
            except KeyError:
                new_string += char
        return new_string.replace('\\', '/')

    output_cmd = []
    commands = ' '.join(input_command.split()).split('|')
    STDOUT = (-1, '')  # -1 is PIPE

    for count, command in enumerate(commands[:-1]):

        assert '>' not in command.split('&')[-1], \
            'You cannot redirect stdout with > and pipe the' \
            ' outputs at the same time:\n\t%s' % command

        # check for stdin
        # -1 means use of stdout from the command before this command
        stdin = -1 if count != 0 else ''
        if '<' in command:
            if count == 0:
                stdin = _raw(shlex.split(command.split('<')[-1])[0])
                # replace stdin from the original command
                command = re.sub(r"\s<+\s\S+", '', command)
            else:
                raise ValueError(
                    'You cannot use < for stdin and pipe data to command '
                    'at the same time:\n\t%s' % command
                )

        output_cmd.append(
            {
                'cmd': command.strip().replace('\\', '/'),
                'stdin': stdin.strip() if stdin != -1 else stdin,
                'stdout': STDOUT
            }
        )

    # set-up stdout for the last command
    last_command = commands[-1]
    mode = None
    if '>>' in last_command:
        mode = 'a'
        stdout = _raw(shlex.split(last_command.split('>>')[-1])[0])
    elif '>' in last_command:
        mode = 'w'
        stdout = _raw(shlex.split(last_command.split('>')[-1])[0])
    else:
        stdout = subprocess.PIPE

    command = re.sub(r"\s>+\s\S+", '', last_command)

    stdin = None if len(commands) == 1 else -1
    if '<' in command:
        # now check for stdin
        stdin = _raw(shlex.split(command.split('<')[-1])[0])
        # replace stdin from the original command
        command = re.sub(r"\s<+\s\S+", '', command)

    output_cmd.append(
        {
            'cmd': command.strip().replace('\\', '/'),
            'stdin': stdin,
            'stdout': (stdout, mode)
        }
    )
    return output_cmd


def _stream_file_content(file_object):
    """Return contents of a file like object as a single string."""
    try:
        file_object.seek(0)
    except AttributeError:
        return str(file_object)
    else:
        return ''.join(map(lambda s: s.decode('utf-8'), file_object.readlines()))
