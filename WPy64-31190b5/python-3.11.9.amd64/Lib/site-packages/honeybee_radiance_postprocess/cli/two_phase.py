"""Commands to work with two phase Radiance matrices using NumPy."""
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


@click.group(help='Commands to work with two phase Radiance matrices using NumPy.')
def two_phase():
    pass


@two_phase.command('rgb-to-illuminance')
@click.argument(
    'total-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'direct-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'direct-sunlight-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--binary/--ascii', is_flag=True, default=True, help='Switch between binary '
    'and ascii input matrices. Default is binary.'
)
@click.option(
    '--float32/--float16', is_flag=True, default=True, help='Switch between float32 '
    'and float16 output matrices. Default is float32.'
)
@click.option(
    '--total-name', '-n', help='Total output file name.', default='total',
    show_default=True
)
@click.option(
    '--direct-name', '-n', help='Direct output file name.', default='direct',
    show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def rgb_to_illuminance(
    total_mtx, direct_mtx, direct_sunlight_mtx, binary, float32, total_name,
    direct_name, output_folder
    ):
    """Process results of two phase simulations (total, direct, direct sunlight).

    The function will replace the direct illuminance with the direct sunlight
    illuminance: total - direct + direct_sunlight.

    The function has two output files. One for the illuminance of the above
    calculation, and one for the direct sunlight illuminance. In both cases
    the conversion from RGB to illuminance is executed.

    \b
    Args:
        total-mtx: Path to total matrix.
        direct-mtx: Path to direct matrix.
        direct-sunlight-mtx: Path to direct sunlight matrix.
    """
    try:
        if binary:
            total = binary_to_array(total_mtx)
            direct = binary_to_array(direct_mtx)
            direct_sunlight = binary_to_array(direct_sunlight_mtx)
        else:
            total = ascii_to_array(total_mtx)
            direct = ascii_to_array(direct_mtx)
            direct_sunlight = ascii_to_array(direct_sunlight_mtx)

        data = total - direct + direct_sunlight

        conversion = np.array([47.4, 119.9, 11.6], dtype=np.float32)
        total_illuminance = np.dot(data, conversion)
        direct_sunlight_illuminance = np.dot(direct_sunlight, conversion)

        if not float32:
            total_illuminance = np.minimum(
                total_illuminance, np.finfo(np.float16).max).astype(
                np.float16)
            direct_sunlight_illuminance = np.minimum(
                direct_sunlight_illuminance, np.finfo(np.float16).max).astype(
                np.float16)

        # save total illuminance
        total_output = Path(output_folder, total_name)
        total_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(total_output, total_illuminance)

        # save direct sunlight illuminance
        direct_output = Path(output_folder, direct_name)
        direct_output.parent.mkdir(parents=True, exist_ok=True)

        np.save(direct_output, direct_sunlight_illuminance)

    except Exception:
        _logger.exception('Processing annual results failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@two_phase.command('rgb-to-illuminance-file')
@click.argument(
    'mtx-file', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--binary/--ascii', is_flag=True, default=True, help='Switch between binary '
    'and ascii input matrices. Default is binary.'
)
@click.option(
    '--float32/--float16', is_flag=True, default=True, help='Switch between float32 '
    'and float16 output matrices. Default is float32.'
)
@click.option(
    '--name', '-n', help='Name of output file.', default='illuminance',
    show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def rgb_to_illuminance_file(
    mtx_file, binary, float32, name, output_folder
    ):
    """Convert a RGB Radiance matrix to illuminance and save the array as a
    NumPy file.

    \b
    Args:
        mtx-file: Path to matrix file to convert.
    """
    try:
        if binary:
            mtx = binary_to_array(mtx_file)
        else:
            mtx = ascii_to_array(mtx_file)

        conversion = np.array([47.4, 119.9, 11.6], dtype=np.float32)
        total_illuminance = np.dot(mtx, conversion)

        if not float32:
            total_illuminance = np.minimum(
                total_illuminance, np.finfo(np.float16).max).astype(
                np.float16)

        # save total illuminance
        total_output = Path(output_folder, name)
        total_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(total_output, total_illuminance)

    except Exception:
        _logger.exception('Processing annual results failed.')
        sys.exit(1)
    else:
        sys.exit(0)


@two_phase.command('add-remove-sky-matrix')
@click.argument(
    'total-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'direct-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.argument(
    'direct-sunlight-mtx', type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--binary/--ascii', is_flag=True, default=True, help='Switch between binary '
    'and ascii input matrices. Default is binary.'
)
@click.option(
    '--float32/--float16', is_flag=True, default=True, help='Switch between float32 '
    'and float16 output matrices. Default is float32.'
)
@click.option(
    '--total-name', '-n', help='Total output file name.', default='total',
    show_default=True
)
@click.option(
    '--direct-name', '-n', help='Direct output file name.', default='direct',
    show_default=True
)
@click.option(
    '--output-folder', '-of', help='Output folder.', default='.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True)
)
def add_remove_sky_matrix(
    total_mtx, direct_mtx, direct_sunlight_mtx, binary, float32, total_name,
    direct_name, output_folder
    ):
    """Process results of two phase simulations (total, direct, direct sunlight).

    The function will replace the direct values with the direct sunlight
    values: total - direct + direct_sunlight.

    The function has two output files. One for the results of the above
    calculation, and one for the direct sunlight values.

    \b
    Args:
        total-mtx: Path to total matrix.
        direct-mtx: Path to direct matrix.
        direct-sunlight-mtx: Path to direct sunlight matrix.
    """
    try:
        if binary:
            total = binary_to_array(total_mtx)
            direct = binary_to_array(direct_mtx)
            direct_sunlight = binary_to_array(direct_sunlight_mtx)
        else:
            total = ascii_to_array(total_mtx)
            direct = ascii_to_array(direct_mtx)
            direct_sunlight = ascii_to_array(direct_sunlight_mtx)

        data = total - direct + direct_sunlight

        if not float32:
            data = np.minimum(
                data, np.finfo(np.float16).max).astype(
                np.float16)
            direct_sunlight = np.minimum(
                direct_sunlight, np.finfo(np.float16).max).astype(
                np.float16)

        # save total values
        total_output = Path(output_folder, total_name)
        total_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(total_output, data)

        # save direct values
        direct_output = Path(output_folder, direct_name)
        direct_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(direct_output, direct_sunlight)

    except Exception:
        _logger.exception('Processing annual results failed.')
        sys.exit(1)
    else:
        sys.exit(0)
