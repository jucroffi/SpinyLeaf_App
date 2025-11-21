"""Commands to work with Radiance matrices using NumPy."""
import sys
import logging
from pathlib import Path
import click
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ..reader import binary_to_array, ascii_to_array

_logger = logging.getLogger(__name__)


@click.group(help='Commands to work with Radiance matrices using NumPy.')
def mtxop():
    pass


@mtxop.command('operate-two')
@click.argument(
    'first-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'second-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--operator', type=click.Choice(['+', '-', '/', '*']), default='+',
    help='Operation between the two matrices.'
)
@click.option(
    '--conversion', help='Conversion as a string. This option is useful to post-process '
    'the results from 3 RGB components into one as part of this command.'
)
@click.option(
    '--binary/--ascii', is_flag=True, default=True, help='Switch between binary '
    'and ascii input matrices. Default is binary.'
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def two_matrix_operations(
    first_mtx, second_mtx, operator, conversion, binary, name, output_folder
        ):
    """Operations between two Radiance matrices.

    The input matrices must be Radiance binary matrices. The operations will be performed
    elementwise for the matrices.

    \b
    Args:
        first-mtx: Path to fist matrix.
        second-mtx: Path to second matrix.
    """
    try:
        if binary:
            first = binary_to_array(first_mtx)
            second = binary_to_array(second_mtx)
        else:
            first = ascii_to_array(first_mtx)
            second = ascii_to_array(second_mtx)

        data = eval('first %s second' % operator)

        if conversion:
            conversion = list(map(float, conversion.split(' ')))
            conversion = np.array(conversion, dtype=np.float32)
            data = np.dot(data, conversion)

        output = Path(output_folder, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, data)

    except Exception:
        _logger.exception('Operation on two Radiance matrix failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@mtxop.command('operate-three')
@click.argument(
    'first-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'second-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'third-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--operator-one', type=click.Choice(['+', '-', '/', '*']), default='+',
    help='Operation between the two matrices.'
)
@click.option(
    '--operator-two', type=click.Choice(['+', '-', '/', '*']), default='+',
    help='Operation between the two matrices.'
)
@click.option(
    '--conversion', help='Conversion as a string. This option is useful to post-process '
    'the results from 3 RGB components into one as part of this command.'
)
@click.option(
    '--binary/--ascii', is_flag=True, default=True, help='Switch between binary '
    'and ascii input matrices. Default is binary.'
)
@click.option(
    '--name', '-n', help='Output file name.', default='output', show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def three_matrix_operations(
    first_mtx, second_mtx, third_mtx, operator_one, operator_two, conversion,
    binary, name, output_folder
    ):
    """Operations between three Radiance matrices.

    The input matrices must be Radiance binary matrices. The operations will be performed
    elementwise for the matrices.

    \b
    Args:
        first-mtx: Path to fist matrix.
        second-mtx: Path to second matrix.
        third-mtx: Path to third matrix.
    """
    try:
        if binary:
            first = binary_to_array(first_mtx)
            second = binary_to_array(second_mtx)
            third = binary_to_array(third_mtx)
        else:
            first = ascii_to_array(first_mtx)
            second = ascii_to_array(second_mtx)
            third = ascii_to_array(third_mtx)

        data = eval('first %s second %s third' % (operator_one, operator_two))

        if conversion:
            conversion = list(map(float, conversion.split(' ')))
            conversion = np.array(conversion, dtype=np.float32)
            data = np.dot(data, conversion)

        output = Path(output_folder, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, data)

    except Exception:
        _logger.exception('Operation on three Radiance matrix failed.')
        sys.exit(1)
    else:
        sys.exit(0)
