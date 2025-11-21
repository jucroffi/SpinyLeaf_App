"""honeybee radiance postprocess grid commands."""
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
from ..annualdaylight import _annual_daylight_vis_metadata

_logger = logging.getLogger(__name__)


@click.group(help='Commands for generating and modifying sensor grids.')
def grid():
    pass


@grid.command('merge-folder')
@click.argument(
    'input-folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.argument(
    'output-folder',
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True))
@click.argument('extension', type=str)
@click.option(
    '--dist-info', '-di',
    help='An optional input for distribution information to put the grids back together '
    '. Alternatively, the command will look for a _redist_info.json file inside the '
    'folder.', type=click.Path(file_okay=True, dir_okay=False, resolve_path=True)
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
def merge_grid_folder(input_folder, output_folder, extension, dist_info,
                      output_extension, as_text, fmt, delimiter):
    """Restructure files in a distributed folder.

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
        restore_original_distribution(
            input_folder, output_folder, extension, dist_info, output_extension,
            as_text, fmt, delimiter)
    except Exception:
        _logger.exception('Failed to restructure data from folder.')
        sys.exit(1)
    else:
        sys.exit(0)


@grid.command('merge-folder-metrics')
@click.argument(
    'input-folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.argument(
    'output-folder',
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True))
@click.option(
    '--dist-info', '-di',
    help='An optional input for distribution information to put the grids back together '
    '. Alternatively, the command will look for a _redist_info.json file inside the '
    'folder.', type=click.Path(file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-info', '-gi',
    help='An optional input for grid information that will be copied to each '
    'metric folder. This file is usually called grids_info.json.',
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True)
)
def merge_metrics_folder(input_folder, output_folder, dist_info, grids_info):
    """Restructure annual daylight metrics in a distributed folder.
    
    Since this command redistributes metrics it is expected that the input
    folder has sub folder

    \b
    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder
    """
    try:
        # handle optional case for Functions input
        if dist_info and not Path(dist_info).is_file():
            dist_info = None
        if grids_info:
            with open(grids_info) as gi:
                grids_info = json.load(gi)
        extension_mapper = {
            'da': 'da',
            'cda': 'cda',
            'udi': 'udi',
            'udi_lower': 'udi',
            'udi_upper': 'udi'
        }
        metric_info_dict = _annual_daylight_vis_metadata()
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        for metric, extension in extension_mapper.items():
            metric_folder = input_folder.joinpath(metric)
            metric_out = output_folder.joinpath(metric)
            restore_original_distribution_metrics(
                metric_folder, output_folder, metric, extension, dist_info)

            if grids_info:
                info_file = metric_out.joinpath('grids_info.json')
                info_file.write_text(json.dumps(grids_info))

            vis_data = metric_info_dict[metric]
            vis_metadata_file = metric_out.joinpath('vis_metadata.json')
            vis_metadata_file.write_text(json.dumps(vis_data, indent=4))
    except Exception:
        _logger.exception('Failed to restructure data from folder.')
        sys.exit(1)
    else:
        sys.exit(0)


def restore_original_distribution(
        input_folder, output_folder, extension='npy', dist_info=None,
        output_extension='ill', as_text=False, fmt='%.2f', input_delimiter=',',
        delimiter='tab'):
    """Restructure files to the original distribution based on the distribution info.
    
    It will assume that the files in the input folder are NumPy files. However,
    if it fails to load the files as arrays it will try to load from binary
    Radiance files to array.

    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder
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
        input_delimiter: Delimiter for the input files. This is used only if the
            input files are text files.
        delimiter: Delimiter for the output files when saved as text.
    """
    if not dist_info:
        _redist_info_file = Path(input_folder, '_redist_info.json')
    else:
        _redist_info_file = Path(dist_info)

    assert _redist_info_file.is_file(), 'Failed to find %s' % _redist_info_file

    with open(_redist_info_file) as inf:
        data = json.load(inf)

    # create output folder
    output_folder = Path(output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True, exist_ok=True)

    src_file = Path()
    for f in data:
        output_file = Path(output_folder, f['identifier'])
        # ensure the new folder is created. in case the identifier has a subfolder
        parent_folder = output_file.parent
        if not parent_folder.is_dir():
            parent_folder.mkdir()

        out_arrays = []
        for src_info in f['dist_info']:
            st = src_info['st_ln']
            end = src_info['end_ln']
            new_file = Path(input_folder, '%s.%s' % (src_info['identifier'], extension))
            if not new_file.samefile(src_file):
                src_file = new_file
                try:
                    array = np.load(src_file)
                except:
                    try:
                        array = binary_to_array(src_file)
                    except:
                        try:
                            array = np.loadtxt(
                                src_file, delimiter=input_delimiter)
                        except Exception:
                            raise RuntimeError(
                                f'Failed to load input file "{src_file}"')
            slice_array = array[st:end+1,:]

            out_arrays.append(slice_array)

        out_array = np.concatenate(out_arrays)
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
            elif delimiter == 'comma':
                delimiter = ','
            np.savetxt(output_file.with_suffix(f'.{output_extension}'),
                       out_array, fmt=fmt, delimiter=delimiter)


def restore_original_distribution_metrics(
        input_folder, output_folder, metric, extension, dist_info=None):
    """Restructure files to the original distribution based on the distribution info.
    
    It will assume that the files in the input folder are NumPy files. However,
    if it fails to load the files as arrays it will try to load from binary
    Radiance files to array.

    Args:
        input_folder: Path to input folder.
        output_folder: Path to the new restructured folder
        metric: Name of the metric to redistribute.
        extension: Extension of the files to collect data from. For annual
            daylight metrics the extension can be 'da', 'cda', or 'udi'.
        dist_info: Path to dist_info.json file. If None, the function will try to load
            ``_redist_info.json`` file from inside the input_folder. (Default: None).
    """
    if not dist_info:
        _redist_info_file = Path(input_folder, '_redist_info.json')
    else:
        _redist_info_file = Path(dist_info)

    assert _redist_info_file.is_file(), 'Failed to find %s' % _redist_info_file

    with open(_redist_info_file) as inf:
        data = json.load(inf)

    # create output folder
    output_folder = Path(output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()

    src_file = Path()
    for f in data:
        output_file = Path(output_folder, metric, '%s.%s' % (f['identifier'], extension))
        # ensure the new folder is created. in case the identifier has a subfolder
        parent_folder = output_file.parent
        if not parent_folder.is_dir():
            parent_folder.mkdir()

        out_arrays = []
        for src_info in f['dist_info']:
            st = src_info['st_ln']
            end = src_info['end_ln']
            new_file = Path(input_folder, '%s.%s' % (src_info['identifier'], extension))
            if not new_file.samefile(src_file):
                src_file = new_file
                array = np.loadtxt(src_file)
            slice_array = array[st:end+1]
            out_arrays.append(slice_array)

        out_array = np.concatenate(out_arrays)
        # save array as txt file
        np.savetxt(output_file, out_array, fmt='%.2f')
