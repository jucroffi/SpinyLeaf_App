"""honeybee radiance postprocess merge commands."""
import click
import sys
import logging
import json
from pathlib import Path
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from honeybee_radiance_postprocess.reader import binary_to_array

_logger = logging.getLogger(__name__)


@click.group(help='Commands for generating and modifying sensor grids.')
def merge():
    pass


@merge.command('merge-files')
@click.argument(
    'input-folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.argument('extension', type=str)
@click.option(
    '--output-file', '-of',
    help='Name of the merged file.', default='results',
    type=click.STRING
)
@click.option(
    '--dist-info', '-di',
    help='An optional input for distribution information to put the grids back together '
    '. Alternatively, the command will look for a _redist_info.json file inside the '
    'folder.', type=click.Path(file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--merge-axis', '-ma',
    help='Merge files along axis.', default='0', show_default=True,
    type=click.Choice(['0', '1', '2']), show_choices=True
)
@click.option(
    '--output-extension', '-oe',
    help='Output file extension. This is only used if as_text is set to True. '
    'Otherwise the output extension will be npy.', default='ill', type=click.STRING
)
@click.option(
    '--as-text', '-at',
    help='Set to True if the output files should be saved as text instead of '
    'NumPy files.', default=False, type=click.BOOL
)
@click.option(
    '--fmt',
    help='Format for the output files when saved as text.', default='%.2f',
    type=click.STRING
)
@click.option(
    '--delimiter',
    help='Delimiter for the output files when saved as text.',
    type=click.Choice(['space', 'tab']), default='tab'
)
def merge_files(
        input_folder, output_file, extension, dist_info, merge_axis,
        output_extension, as_text, fmt, delimiter):
    """Merge files in a distributed folder.

    \b
    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder
        extension: Extension of the files to collect data from. It will be ``pts`` for
            sensor files. Another common extension is ``ill`` for the results of daylight
            studies.
    """
    try:
        # handle optional case for Functions input
        if dist_info and not Path(dist_info).is_file():
            dist_info = None
        _merge_files(input_folder, output_file, int(merge_axis), extension, dist_info,
                    output_extension, as_text, fmt, delimiter)
    except Exception:
        _logger.exception('Failed to merge files from folder.')
        sys.exit(1)
    else:
        sys.exit(0)


def _merge_files(
        input_folder, output_file, merge_axis=0, extension='npy', dist_info=None,
        output_extension='ill', as_text=False, fmt='%.2f', delimiter='tab'):
    """Restructure files to the original distribution based on the distribution info.

    It will assume that the files in the input folder are NumPy files. However,
    if it fails to load the files as arrays it will try to load from binary
    Radiance files to array.

    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder.
        merge_axis: Merge along axis.
        extension: Extension of the files to collect data from. Default is ``npy`` for
            NumPy files. Another common extension is ``ill`` for the results of daylight
            studies.
        dist_info: Path to dist_info.json file. If None, the function will try to load
            ``_redist_info.json`` file from inside the input_folder. (Default: None).
        output_extension: Output file extension. This is only used if as_text
            is set to True. Otherwise the output extension will be ```npy``.
        as_text: Set to True if the output files should be saved as text instead
            of NumPy files.
        fmt: Format for the output files when saved as text.
        delimiter: Delimiter for the output files when saved as text.
    """
    if not dist_info:
        _redist_info_file = Path(input_folder, '_redist_info.json')
    else:
        _redist_info_file = Path(dist_info)

    assert _redist_info_file.is_file(), 'Failed to find %s' % _redist_info_file

    with open(_redist_info_file) as inf:
        data = json.load(inf)

    out_arrays = []
    src_file = Path()
    for f in data:
        output_file = Path(output_file)
        # ensure the new folder is created. in case the identifier has a subfolder
        parent_folder = output_file.parent
        if not parent_folder.is_dir():
            parent_folder.mkdir()

        for src_info in f['dist_info']:
            new_file = Path(input_folder, '%s.%s' %
                            (src_info['identifier'], extension))

            if not new_file.samefile(src_file):
                src_file = new_file
                try:
                    array = np.load(src_file)
                except Exception:
                    array = binary_to_array(src_file)

                out_arrays.append(array)

        out_array = np.concatenate(out_arrays, axis=merge_axis)

        # save numpy array, .npy extension is added automatically
        if not as_text:
            np.save(output_file, out_array)
        else:
            if output_extension.startswith('.'):
                output_extension = output_extension[1:]
            if delimiter == 'tab':
                delimiter = '\t'
            elif delimiter == 'space':
                delimiter = ' '
            np.savetxt(output_file.with_suffix(f'.{output_extension}'),
                       out_array, fmt=fmt, delimiter=delimiter)
