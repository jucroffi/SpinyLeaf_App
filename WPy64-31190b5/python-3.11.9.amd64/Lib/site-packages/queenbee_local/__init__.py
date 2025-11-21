"""Utility Luigi object for Queenbee local execution.

QueenbeeTask is a wrapper around luigi.Task which is used by queenbee-local to
execute Queenbee-specific task. You probably don't need to use this module in your code.
"""
import logging
import logging.config
import subprocess
import socket
import pathlib
import os
import time
import shutil
import tempfile
import platform
import json
import warnings
import re
from datetime import datetime, timedelta

import luigi

# importing all the methods to make it easy for the extensions to grab them all from here
from .helper import parse_input_args, to_snake_case, update_params, _change_permission, \
    _copy_artifacts, to_camel_case, tab_forward


SYSTEM = platform.system()


class QueenbeeTask(luigi.Task):

    """
    A Luigi task to run a single command as a subprocess.

    Luigi has a subclass for running external programs

    https://luigi.readthedocs.io/en/stable/api/luigi.contrib.external_program.html
    https://github.com/spotify/luigi/blob/master/luigi/contrib/external_program.py

    But:
        * It doesn't allow setting up the environment for the command
        * It doesn't support piping
        * It doesn't support progress reporting

    QueenbeeTask:

        * simplifies the process of writing commands
        * captures stdin / stdout
        * supports piping
    """

    def _get_log_folder(self, folder) -> str:
        """A hack to find the log file!"""
        folder = pathlib.Path(folder)
        log_file = folder.joinpath('__logs__', 'logs.cfg')
        if log_file.exists():
            return log_file.as_posix()
        else:
            return self._get_log_folder(folder.parent)

    def status_file(self) -> pathlib.Path:
        folder = pathlib.Path(self._get_log_folder(self.initiation_folder)).parent
        # the empty status file should be created by queenbee-luigi
        status_file = folder.joinpath('status.json')
        if not status_file.is_file():
            status_file.write_text(
                json.dumps({
                    'meta': {
                        'progress': {
                            'completed': 0,
                            'running': 0,
                            'total': 0
                        },
                        'resources_duration': {
                            'cpu': 0,
                            'memory': 0
                        }
                    }
                })
            )
        return status_file

    @property
    def log_config(self) -> str:
        return self._get_log_folder(self.initiation_folder)

    @property
    def task_image(self):
        raise NotImplementedError('`task_image` should be overwritten by each task.')

    @property
    def image_workdir(self):
        raise NotImplementedError('`image_workdir` should be overwritten by each task.')

    def get_interface_logger(self):
        logger = logging.getLogger('luigi-interface')
        if not logger.handlers:
            # load the logs from shared config file
            # This is an issue in Windows with multiple processors
            # https://github.com/spotify/luigi/issues/2247
            logging.config.fileConfig(self.log_config)
        return logger

    def get_queenbee_logger(self):
        logger = logging.getLogger('queenbee-interface')
        if not logger.handlers:
            logging.config.fileConfig(self.log_config)
        return logger

    def get_queenbee_errors_logger(self):
        logger = logging.getLogger('queenbee-error')
        if not logger.handlers:
            logging.config.fileConfig(self.log_config)
        return logger

    @property
    def input_artifacts(self):
        """Task's input artifacts.

        These artifacts will be copied to execution folder before executing the command.
        """
        return []

    @property
    def output_artifacts(self):
        """Task's output artifacts.

        These artifacts will be copied to study folder from execution folder after the
        task is done.
        """
        return []

    @property
    def output_parameters(self):
        """Task's output parameters.

        These parameters will be copied to study folder from execution folder after the
        task is done.
        """
        return []

    def command(self):
        """An executable command which will be passed to subprocess.Popen

        Overwrite this method.
        """
        raise NotImplementedError(
            'Command method must be overwritten in every subclass.'
        )

    def _copy_script(self, dst):
        """Render and copy the template script to dst folder.

        Args:
            dst: Execution folder.
        """
        logger = self.get_interface_logger()
        logger.info(f"{self.__class__.__name__}: started copying script.")
        script_file = self.__script__
        data = {
            f'inputs.{k.replace("_", "-")}': v
            for k, v in self.input_parameters.items()
        }
        pattern = r'[^{\{]+(?=}\})'
        script_content = script_file.read_text()
        matches = re.finditer(pattern, script_content, re.MULTILINE)
        for match in matches:
            place_holder = match.group()
            script_content = script_content.replace(
                f'{{{{{place_holder}}}}}', str(data[place_holder.strip()])
            )
        out_file = pathlib.Path(dst, '__scripts__', 'script.py')
        out_file.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        out_file.write_text(script_content)
        logger.info(f"{self.__class__.__name__}: finished copying script.")

    def _copy_input_artifacts(self, dst):
        """Copy input artifacts to destination folder.

        Args:
            dst: Execution folder.
        """
        logger = self.get_interface_logger()
        for art in self.input_artifacts:
            logger.info(
                f"{self.__class__.__name__}: copying input artifact {art['name']} from "
                f"{art['from']} ..."
            )
            is_optional = art.get('optional', False)
            try:
                _copy_artifacts(art['from'], os.path.join(dst, art['to']), is_optional)
            except TypeError as e:
                if is_optional:
                    continue
                raise TypeError(
                    f'Failed to copy input artifact: {art["name"]}\n{e}'
                )
        logger.info(
            f"{self.__class__.__name__}: finished copying artifacts..."
        )

    def _create_optional_output_artifacts(self, dst):
        """Create a dummy place holder for optional output artifacts.

        Args:
            dst: Execution folder
        """
        logger = self.get_interface_logger()
        for art in self.output_artifacts:
            is_optional = art.get('optional', False)
            if not is_optional:
                continue
            artifact = pathlib.Path(dst, art['from'])
            if artifact.exists():
                continue
            if 'type' not in art:
                logger.exception(
                    f"{self.__class__.__name__}: Optional artifact {art['name']} is "
                    "missing type key. Try to regenerate the recipe with a newer "
                    "version of queenbee-luigi."
                )
            output_type = art['type']
            logger.info(
                f"{self.__class__.__name__}: creating an empty {output_type} for "
                f"optional artifact {art['name']} at {artifact} ..."
            )
            if output_type == 'folder':
                artifact.mkdir(parents=True, exist_ok=True)
            else:
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text('')

    def _copy_output_artifacts(self, src, raise_on_error=True):
        """Copy output artifacts to project folder.

        Args:
            src: Execution folder.
        """
        logger = self.get_interface_logger()
        for art in self.output_artifacts:
            logger.info(
                f"{self.__class__.__name__}: copying output artifact {art['name']} "
                f"to {art['to']} ..."
            )
            is_optional = art.get('optional', False)
            try:
                _copy_artifacts(os.path.join(src, art['from']), art['to'], is_optional)
            except Exception:
                if is_optional:
                    continue
                if not raise_on_error:
                    continue
                logger.exception(
                    f"Failed to copy output artifact: {art['name']} to {art['to']} ..."
                )

    def _copy_output_parameters(self, src):
        """Copy output parameters to project folder.

        Args:
            src: Execution folder.
        """
        logger = self.get_interface_logger()
        for art in self.output_parameters:
            logger.info(
                f"{self.__class__.__name__}: copying output parameters {art['name']} "
                f"to {art['to']} ..."
            )
            _copy_artifacts(os.path.join(src, art['from']), art['to'])

    @property
    def _is_debug_mode(self):
        if '__debug__' not in self._input_params:
            return False
        return self._input_params['__debug__']

    @property
    def _runner(self):
        return self._input_params.get('__runner__', 'System').lower()

    def _get_dst_folder(self, command):
        debug_folder = self._is_debug_mode
        dst_dir = tempfile.mkdtemp(prefix=f'{self.__class__.__name__}-')
        if debug_folder:
            dst_dir = os.path.join(debug_folder, os.path.split(dst_dir)[-1])
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)

            if SYSTEM == 'Windows':
                file_name = 'command.bat'
                content = '%s\npause' % command
            else:
                file_name = 'command.sh'
                content = '#!/bin/bash\nfunction pause(){\n\tread -p "$*"\n}' \
                    '\n%s\npause \'Press [Enter] key to continue...\'\n' % command

            command_file = os.path.join(dst_dir, file_name)
            with open(command_file, 'w') as outf:
                outf.write(content)

            os.chmod(command_file, 0o777)
        return dst_dir

    def run(self):
        st_time = datetime.now()
        logger = self.get_interface_logger()
        qb_logger = self.get_queenbee_logger()
        err_logger = self.get_queenbee_errors_logger()
        # replace ' with " for Windows systems and vise versa for unix systems
        command = self.command()
        if SYSTEM == 'Windows':
            command = command.replace('\'', '"')
        else:
            command = command.replace('"', '\'')

        cur_dir = os.getcwd()
        dst = self._get_dst_folder(command)
        os.chdir(dst)

        self._copy_input_artifacts(dst)
        if hasattr(self, 'is_script') and self.is_script:
            self._copy_script(dst)

        logger.info(f'Started running {self.__class__.__name__}...')
        qb_logger.info(f'Started running {self.__class__.__name__}...')
        self._update_status(logger=qb_logger, started=True)

        if self._runner == 'wsl':
            command = f'wsl {command}'
        elif self._runner in ['docker', 'podman']:
            dst_path = pathlib.Path(dst)
            if hasattr(self, 'task_image'):
                command = f'{self._runner} run -it --rm ' \
                    f'--volume {dst_path.as_posix()}:/home/temp/{dst_path.name} ' \
                    f'-w /home/temp/{dst_path.name} ' \
                    f'{self.task_image} {command}'
            else:
                message = 'This recipe is generated by an older version of ' \
                    'queenbee-luigi and cannot be executed in a container.'
                logger.info(message)
                qb_logger.info(message)

        p = subprocess.Popen(
            command, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, shell=True, env=os.environ
        )

        msg = ''
        for line in iter(p.stdout.readline, b''):
            try:
                msg = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                msg = line.decode('latin-1').strip()
            logger.info(msg)
            qb_logger.info(msg)

        p.communicate()

        if p.returncode != 0:
            # try to copy artifacts if they have been created. This can be helpful if
            # the artifact has been generated and the information are helpful for
            # debugging.
            self._copy_output_artifacts(dst, raise_on_error=False)
            err_msg = f'"{self.__class__.__name__}" failed. See below for more ' \
                f'information:\n\n    {msg}\n'
            qb_logger.error(err_msg)
            logger.error(err_msg)
            err_logger.error(err_msg)
            self._update_status(logger=qb_logger, started=False)
            raise ValueError(err_msg)

        # copy the results file back
        self._create_optional_output_artifacts(dst)
        self._copy_output_artifacts(dst)
        self._copy_output_parameters(dst)
        # change back to initial directory
        os.chdir(cur_dir)
        # delete the temp folder content
        try:
            shutil.rmtree(dst, ignore_errors=True)
        except Exception:
            # folder is in use or running in debug mode
            # this is a temp folder so not deleting is not a major issue
            pass
        duration = datetime.now() - st_time
        duration -= timedelta(microseconds=duration.microseconds)  # remove microseconds
        logger.info(f'...finished running {self.__class__.__name__} in {duration}')
        qb_logger.info(f'...finished running {self.__class__.__name__} in {duration}')
        self._update_status(
            logger=qb_logger, started=False, cpu_seconds=duration.seconds
        )

    @staticmethod
    def load_input_param(input_param):
        """A static class kept here for backwards compatibility.

        Use the `load_input_param` function directly instead.
        """
        warnings.warn(
            'load_input_param classmethod is deprecated. Update your code to use the '
            'function directly instead.',
            category=DeprecationWarning, stacklevel=2
        )
        return load_input_param(input_param)

    def _try_update_status(self, logger, started, cpu_seconds, sleep=0):
        if sleep:
            logger.info(
                f'Re-trying to read and update the status after {sleep} seconds.'
            )
            time.sleep(sleep)

        with self._status_lock:
            # open JSON file
            inf = self.status_file()
            # update values
            try:
                data = json.loads(inf.read_text())
            except json.JSONDecodeError:
                # this should not happen unless user directly interacts with the status
                # file and blocks the status update
                logger.info('Failed to read the status file.')
                # create a copy for debugging.
                shutil.copy(inf.as_posix(), inf.parent.joinpath(inf.name + 'bak'))
                return False
            else:
                data['meta']['resources_duration']['cpu'] += cpu_seconds
                progress = data['meta']['progress']
                if started:
                    progress['running'] += 1
                    progress['total'] += 1
                else:
                    # task finished
                    progress['running'] -= 1
                    progress['completed'] += 1
                    if progress['running'] < 0:
                        # apparently this can happen is complicated workflows.
                        # I can spend hours to replicate and fix this issue but it is not
                        # really critical. This small fix will make it look reasonable.
                        progress['running'] = 0

                # write the json file
                logger.info(
                    f'[{progress["completed"]}/{progress["total"]}] completed. '
                    f'{progress["running"]} running.'
                )
                try:
                    inf.write_text(json.dumps(data))
                except PermissionError:
                    logger.info('PermissionError: Failed to write the updated status file.')
                    return False
                return True

    def _update_status(self, logger, started=True, cpu_seconds=0):
        """Update run status."""
        if not hasattr(self, '_status_lock'):
            warnings.warn(
                'This recipe is generated by an older version of queenbee luigi. '
                'Update the recipe to get the run status.'
            )
            return

        for i in range(3):
            success = self._try_update_status(
                logger=logger, started=started, cpu_seconds=cpu_seconds, sleep=i * 3
            )
            if success:
                return
            else:
                # waite for the permission issue to resolve
                time.sleep((i + 1) * 0.5)


def load_input_param(input_param):
    """This function tries to import the values from a file as a Task input
        parameter.

    It first tries to import the content as a dictionary assuming the input file is
    a JSON file. If the import as JSON fails it will import the content as string and
    split them by next line. If there are several items it will return a list,
    otherwise it will return a single item.
    """
    content = ''
    with open(input_param, 'r') as param:
        try:
            content = json.load(param)
        except json.decoder.JSONDecodeError:
            # not a JSON file
            pass
        else:
            return content
    with open(input_param, 'r') as param:
        content = param.read().splitlines()
        if len(content) == 1:
            content = content[0]
    return content


def local_scheduler():
    """Check if luigi Daemon is running.

    If it does then return False otherwise return True.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', 8082))
    except Exception:
        # luigi is running
        local_schedule = False
    else:
        # print('Using local scheduler')
        local_schedule = True
    finally:
        sock.close()
        return local_schedule


LOGS_CONFIG = """
[formatters]
keys: default

[loggers]
keys: root, luigi-interface, queenbee-interface, queenbee-error

[handlers]
keys: console, logfile, errfile

[formatter_default]
format: %(asctime)s %(levelname)s: %(message)s
datefmt:%Y-%m-%d %H:%M:%S

[handler_console]
class: StreamHandler
args: [sys.stdout,]
formatter: default
level: INFO

[handler_logfile]
class: FileHandler
args: ['WORKFLOW.LOG',]
formatter: default
level: DEBUG

[handler_errfile]
class: FileHandler
args: ['ERROR.LOG',]
formatter: default
level: ERROR

[logger_root]
handlers: errfile, logfile
qualname: root
propagate=0

[logger_luigi-interface]
handlers: logfile
qualname: luigi-interface
propagate=0
level: DEBUG

[logger_queenbee-error]
handlers: errfile
qualname: queenbee-error
propagate=0
level: ERROR

[logger_queenbee-interface]
handlers: console
qualname: queenbee-interface
propagate=0
level: INFO

"""
