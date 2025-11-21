import click

import sys
import logging
import os
import json
import re

from honeybee_radiance_folder.folder import ModelFolder as Folder
from honeybee_radiance_folder.folderutil import _as_posix, add_output_spec_to_receiver


# command group for all radiance extension commands.
@click.group(help='honeybee folder commands.')
@click.version_option()
def folder():
    pass


_logger = logging.getLogger(__name__)


@folder.command('aperture-files')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '--default/--black', is_flag=True, default=True, show_default=True,
    help='Flag to note whether black material files must be returned.'
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def aperture_files(radiance_folder, model, default, log_file):
    """Return aperture files."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        black_out = not default
        aperture_files = folder.aperture_files(black_out=black_out)
        log_file.write(json.dumps(aperture_files))
    except Exception as e:
        _logger.exception('Failed to retrieve aperture files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('scene-files')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '--default/--black', is_flag=True, default=True, show_default=True,
    help='Flag to note whether black material files must be returned.'
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def scene_files(radiance_folder, model, default, log_file):
    """Return scene files."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        black_out = not default
        scene_files = folder.scene_files(black_out=black_out)
        log_file.write(json.dumps(scene_files))
    except Exception as e:
        _logger.exception('Failed to retrieve scene files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('grid-files')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def grid_files(radiance_folder, model, log_file):
    """Return grid files."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        grid_files = folder.grid_files()
        log_file.write(json.dumps(grid_files))
    except Exception as e:
        _logger.exception('Failed to retrieve grid files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('grid-info-files')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def grid_info_files(radiance_folder, model, log_file):
    """Return grid information files."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        grid_info_files = folder.grid_info_files()
        log_file.write(json.dumps(grid_info_files))
    except Exception as e:
        _logger.exception('Failed to retrieve grid information files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('view-files')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def view_files(radiance_folder, model, log_file):
    """Return view files."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        view_files = folder.view_files()
        log_file.write(json.dumps(view_files))
    except Exception as e:
        _logger.exception('Failed to retrieve view files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('view-info-files')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def view_info_files(radiance_folder, model, log_file):
    """Return view information files."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        view_info_files = folder.view_info_files()
        log_file.write(json.dumps(view_info_files))
    except Exception as e:
        _logger.exception('Failed to retrieve view information files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('aperture-groups')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '--exterior/--interior', is_flag=True, default=True, show_default=True,
    help='Flag to note whether interior or exterior aperture groups must be returned.'
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def aperture_groups(radiance_folder, model, exterior, log_file):
    """Return aperture groups."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        aperture_groups = folder.aperture_groups(not exterior, reload=True)
        log_file.write(json.dumps(aperture_groups))
    except Exception as e:
        _logger.exception('Failed to retrieve view files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('add-output-spec')
@click.argument(
    'receiver-file', type=click.Path(exists=True, file_okay=True, resolve_path=False)
)
@click.argument('output-spec', type=click.STRING)
@click.option(
    '-o', '--output-file', help='Optional path for an output file. By default this '
    'command modifies the input receiver file.', default=None, show_default=True
)
def add_output_spec(receiver_file, output_spec, output_file):
    """Add output spec to a receiver file.
    
    Args:
        receiver_file: Path to a receiver file. You can find these files under the
            ``aperture_group`` subfolder.
        output_spec: A string for receiver output spec.
    """
    try:
        add_output_spec_to_receiver(
            receiver_file=receiver_file, output_spec=output_spec,
            output_file=output_file
        )
    except Exception as e:
        _logger.exception('Failed to add output spec to receiver.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('dynamic-scene')
@click.argument(
    'radiance-folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '-m', '--model', help='Optional name for model folder', default='model',
    show_default=True
)
@click.option(
    '--outdoor/--indoor', is_flag=True, default=True, show_default=True,
    help='Flag to note whether indoor or outdoor dynamic scene must be returned.'
)
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def dynamic_scene(radiance_folder, model, outdoor, log_file):
    """Return dynamic scene."""
    try:
        folder = Folder(radiance_folder, model_folder=model)
        dynamic_scene = folder.dynamic_scene(not outdoor, reload=True)
        log_file.write(json.dumps(dynamic_scene))
    except Exception as e:
        _logger.exception('Failed to retrieve view files.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('filter-folder')
@click.argument(
    'folder', type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.argument('pattern')
@click.option(
    '-lf', '--log-file', help='Optional log file to export the output. By default this '
    'will be printed to stdout', type=click.File('w'), default='-'
)
def filter_folder(folder, pattern, log_file):
    """Filter files in a folder.

    \b
    Args:
        folder: path to folder.
        pattern: a regex pattern to filter files based on. The pattern must be a raw
            string with no quotes. The command compiles the string to a regex pattern.

    """
    pattern = re.compile("{0}".format(pattern))
    try:
        files = [
            _as_posix(os.path.normpath(os.path.join(folder, f)))
            for f in os.listdir(folder)
            if re.search(pattern, f)
        ]
        log_file.write(json.dumps(files))
    except Exception as e:
        _logger.exception('Failed to filter files inside folder.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


@folder.command('filter-file')
@click.argument(
    'input-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.argument('pattern', default='*:*')
@click.option('--output-file', '-of', type=click.File(mode='w'), default='-')
@click.option(
    '--keep/--remove', is_flag=True, help='A flag to switch between keeping the objects '
    'or removing them from the input list.', default=True, show_default=True
)
def filter_json_file(input_file, pattern, output_file, keep):
    """Filter a list fo JSON objects based on value for a specific key.

    \b
    Args:
        input-file: Path to input JSON file. Input JSON file should be an array of
            JSON objects.
        pattern: Two string values separated by a ``:``. For example group:daylight
            will keep/remove the objects when the value for group key is set to daylight.

    """
    try:
        key, value = [v.strip() for v in pattern.split(':')]

        with open(input_file) as inf:
            data = json.load(inf)

        if key == value == '*':
            # no filtering. pass the values as is.
            output_file.write(json.dumps(data))
            sys.exit(0)
        if keep:
            filtered_data = [obj for obj in data if obj[key] == value]
        else:
            filtered_data = [obj for obj in data if obj[key] != value]

        output_file.write(json.dumps(filtered_data))
    except Exception as e:
        _logger.exception('Failed to filter objects in input file.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)
