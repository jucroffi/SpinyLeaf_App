"""honeybee-openstudio translation commands."""
import sys
import logging
import click

from ladybug.commandutil import process_content_to_output
from honeybee.model import Model

import openstudio
from honeybee_openstudio.writer import model_to_openstudio

_logger = logging.getLogger(__name__)


@click.group(help='Commands for translating between Honeybee and OpenStudio.')
def translate():
    pass


@translate.command('model-to-osm')
@click.argument('model-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--output-file', '-o', help='Optional OSM file path to output the OSM string '
    'of the translation. By default this will be printed to stdout.',
    type=click.File('w'), default='-', show_default=True)
def model_to_osm_cli(model_file, output_file):
    """Translate a Honeybee Model to an OSM file.

    \b
    Args:
        model_file: Full path to a Honeybee Model file (HBJSON or HBpkl).
    """
    try:
        model_to_osm(model_file, output_file)
    except Exception as e:
        _logger.exception(f'Model translation failed:\n{e}')
        sys.exit(1)
    else:
        sys.exit(0)


def model_to_osm(model_file, output_file=None):
    """Translate a Honeybee Model to an OSM file.

    Args:
        model_file: Full path to a Honeybee Model file (HBJSON or HBpkl).
        output_file: Optional OSM file path to output the OSM string of the
            translation. If None, the string will be returned from this function.
    """
    model = Model.from_file(model_file)
    os_model = model_to_openstudio(model, print_progress=True)
    if output_file is not None and 'stdout' not in str(output_file):
        output_file = output_file.name \
            if not isinstance(output_file, str) else output_file
        os_model.save(output_file, overwrite=True)
    else:
        output = process_content_to_output(str(os_model), output_file)
        return output


@translate.command('osm-to-idf')
@click.argument('osm-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--output-file', '-o', help='Optional IDF file path to output the IDF string '
    'of the translation. By default this will be printed to stdout.',
    type=click.File('w'), default='-', show_default=True)
def osm_to_idf_cli(osm_file, output_file):
    """Translate a OSM file to an IDF file.

    \b
    Args:
        osm_file: Full path to a OpenStudio Model file (OSM).
    """
    try:
        osm_to_idf(osm_file, output_file)
    except Exception as e:
        _logger.exception(f'Model translation failed:\n{e}')
        sys.exit(1)
    else:
        sys.exit(0)


def osm_to_idf(osm_file, output_file=None):
    """Translate a Honeybee Model to an OSM file.

    Args:
        osm_file: Full path to an OpenStudio Model file (OSM).
        output_file: Optional IDF file path to output the IDF string of the
            translation. If None, the string will be returned from this function.
    """
    # load the model object from the OSM file
    exist_os_model = openstudio.model.Model.load(osm_file)
    if exist_os_model.is_initialized():
        os_model = exist_os_model.get()
    else:
        raise ValueError(
            'The file at "{}" does not appear to be an OpenStudio model.'.format(
                osm_file
            ))

    # translate the OpenStudio model to an IDF file
    idf_translator = openstudio.energyplus.ForwardTranslator()
    workspace = idf_translator.translateModel(os_model)

    # write the IDF file
    if output_file is not None and 'stdout' not in str(output_file):
        output_file = output_file.name \
            if not isinstance(output_file, str) else output_file
        workspace.save(output_file, overwrite=True)
    else:
        output = process_content_to_output(str(workspace), output_file)
        return output


@translate.command('append-to-osm')
@click.argument('osm-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('model-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--output-file', '-o', help='Optional OSM file path to output the OSM string '
    'of the translation. By default this will be printed to stdout.',
    type=click.File('w'), default='-', show_default=True)
def append_to_osm_cli(osm_file, model_file, output_file):
    """Append a Honeybee Model to a OSM file.

    \b
    Args:
        osm_file: Full path to a OpenStudio Model file (OSM).
        model_file: Full path to a Honeybee Model file (HBJSON or HBpkl).
    """
    try:
        append_to_osm(osm_file, model_file, output_file)
    except Exception as e:
        _logger.exception(f'Model appending failed:\n{e}')
        sys.exit(1)
    else:
        sys.exit(0)


def append_to_osm(osm_file, model_file, output_file=None):
    """Append a Honeybee Model to a OSM file.

    Args:
        osm_file: Full path to an OpenStudio Model file (OSM).
        model_file: Full path to a Honeybee Model file (HBJSON or HBpkl).
        output_file: Optional IDF file path to output the IDF string of the
            translation. If None, the string will be returned from this function.
    """
    # load the model object from the OSM file
    exist_os_model = openstudio.model.Model.load(osm_file)
    if exist_os_model.is_initialized():
        os_model = exist_os_model.get()
    else:
        raise ValueError(
            'The file at "{}" does not appear to be an OpenStudio model.'.format(
                osm_file
            ))

    # load the honeybee Model object
    model = Model.from_file(model_file)
    # append the honeybee model to the OSM
    os_model = model_to_openstudio(model, os_model)

    # write the OSM file
    if output_file is not None and 'stdout' not in str(output_file):
        output_file = output_file.name \
            if not isinstance(output_file, str) else output_file
        os_model.save(output_file, overwrite=True)
    else:
        output = process_content_to_output(str(os_model), output_file)
        return output
