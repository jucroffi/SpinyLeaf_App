"""Commands to translate objects."""
import sys
import logging
from pathlib import Path
import shutil
import click
import json
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ..reader import binary_to_array
from ..util import get_delimiter

_logger = logging.getLogger(__name__)


@click.group(help='Commands to translate objects.')
def translate():
    pass


@translate.command('npy-to-txt')
@click.argument(
    'npy-file', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--extension', '-ext', help='Output file extension', default='txt', show_default=True
)
@click.option(
    '--output-format', '-fmt', help='Output format for each element in the array',
    default='%.7e', show_default=True
)
@click.option(
    '--delimiter', '-d', help='Delimiter in the text file.', default='\t',
    type=click.Choice(['\t', ' ', ',', ';', 'tab', 'space', 'comma', 'semicolon'])
)
def npy_to_txt(npy_file, name, output_folder, extension, output_format, delimiter):
    """Convert a npy file to text file.

    This command reads a NumPy array from a npy file and saves it as readable file. The
    converted file is tab separated.

    \b
    Args:
        npy-file: Path to npy file.
    """
    try:
        delimiter = get_delimiter(delimiter)
        array = np.load(npy_file)
        output = Path(output_folder, f'{name}.{extension}')
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, array, fmt=output_format, delimiter=delimiter)

    except Exception:
        _logger.exception('Converting npy file to text file failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@translate.command('txt-to-npy')
@click.argument(
    'txt-file', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--delimiter', '-d', help='Delimiter in the text file.', default='\t',
    type=click.Choice(['\t', ' ', ',', ';', 'tab', 'space', 'comma', 'semicolon'])
)
def txt_to_npy(txt_file, name, output_folder, delimiter):
    """Convert a text file to npy file.

    This command reads a separated text file and saves it as a NumPy file. As
    an example the input file could be the annual illuminance values.

    \b
    Args:
        txt-file: Path to text file.
    """
    try:
        delimiter = get_delimiter(delimiter)
        array = np.genfromtxt(txt_file, dtype=np.float32, delimiter=delimiter)
        output = Path(output_folder, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, array)

    except Exception:
        _logger.exception('Converting text file to npy file failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@translate.command('binary-to-npy')
@click.argument(
    'mtx-file', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--conversion', help='Conversion as a string. This option is useful to post-process '
    'the results from 3 RGB components into one as part of this command.'
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def binary_to_npy(mtx_file, conversion, name, output_folder):
    """Convert a binary Radiance file to a npy file.

    This command reads a binary Radiance matrix file and saves it as a NumPy file.

    \b
    Args:
        mtx-file: Path to binary Radiance file.
    """
    try:
        array = binary_to_array(mtx_file)
        if conversion:
            conversion = list(map(float, conversion.split(' ')))
            conversion = np.array(conversion, dtype=np.float32)
            array = np.dot(array, conversion)
        output = Path(output_folder, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, array)

    except Exception:
        _logger.exception('Converting binary Radiance file to npy file failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@translate.command('annual-daylight-npy-to-ill')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option(
    '--output-folder', '-of', help='Output folder. If not provided the output '
    'folder will be created in the same directory as the results folder. The '
    'new folder will be called results_ill.', default=None,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def annual_daylight_npy_to_ill(folder, output_folder):
    """Convert an annual daylight results folder to older version.

    This command reads an annual daylight results folder with results saved as
    npy files (NumPy), and converts the npy files to text files in the old
    results folder format.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual
            daylight recipe. Folder should include grids_info.json and
            sun-up-hours.txt.
    """
    try:
        folder = Path(folder)
        static_ill_folder = folder.joinpath('__static_apertures__/default/total')
        if not static_ill_folder.exists():
            raise FileNotFoundError(
                'No results were found for static apertures in the results '
                'folder.')
        grids_info_file = folder.joinpath('grids_info.json')
        if not grids_info_file.exists():
            raise FileNotFoundError(
                'The file grids_info.json was not found in the results folder.')
        sun_up_hours_file = folder.joinpath('sun-up-hours.txt')
        if not sun_up_hours_file.exists():
            raise FileNotFoundError(
                'The file sun-up-hours.txt was not found in the results folder.')

        if output_folder is None:
            output_folder = folder.parent.joinpath('results_ill')
        else:
            output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        with open(grids_info_file) as json_file:
            grids_info = json.load(json_file)

        for grid_info in grids_info:
            full_id = grid_info['full_id']
            npy_file = static_ill_folder.joinpath(f'{full_id}.npy')

            array = np.load(npy_file)
            output = Path(output_folder, full_id + '.ill')
            np.savetxt(output, array, fmt='%.7e', delimiter='\t')

        # copy grids_info and sun-up-hours
        shutil.copy(grids_info_file, output_folder.joinpath('grids_info.json'))
        shutil.copy(sun_up_hours_file, output_folder.joinpath('sun-up-hours.txt'))
    except Exception:
        _logger.exception('Converting annual daylight results folder failed.')
        sys.exit(1)
    else:
        sys.exit(0)
