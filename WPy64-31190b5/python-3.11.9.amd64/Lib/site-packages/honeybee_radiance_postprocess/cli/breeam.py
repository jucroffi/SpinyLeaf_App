"""Commands for BREEAM post-processing."""
import sys
import logging
from pathlib import Path
import os
import json
import click

from ladybug.color import Color
from ladybug.datatype.generic import GenericType
from ladybug.legend import LegendParameters

from honeybee_radiance_postprocess.breeam.breeam import breeam_daylight_assessment_4b

_logger = logging.getLogger(__name__)


@click.group(help='Commands for BREEAM post-processing of Radiance results.')
def breeam():
    pass


@breeam.command('breeam-4b')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.option('--model-file', '-m', help='A Honeybee Model file that was used '
    'in the simulation.', type=click.Path(
    exists=False, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Relative path for subfolder to write output '
    'files.', default='breeam_summary', type=click.Path(
    exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path)
)
def breeam_4b(
    folder, model_file, grids_filter, sub_folder
):
    """Calculate metrics for BREEAM.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe.
    """
    try:
        breeam_daylight_assessment_4b(
            folder, model=model_file, grids_filter=grids_filter,
            sub_folder=sub_folder)
    except Exception:
        _logger.exception('Failed to calculate BREEAM metrics.')
        sys.exit(1)
    else:
        sys.exit(0)


@breeam.command('breeam-4b-vis-metadata')
@click.option(
    '--output-folder', '-o', help='Output folder for vis metadata files.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    default='visualization', show_default=True
)
def breeam_4b_vis(output_folder):
    """Write visualization metadata files for BREEAM 4b."""
    colors = [Color(220, 0, 0), Color(220, 110, 25), Color(255, 190, 0), Color(0, 220, 0)]
    pass_fail_lpar = \
        LegendParameters(min=0, max=3, colors=colors, segment_count=4, title='Pass/Fail')
    pass_fail_lpar.ordinal_dictionary = {
        0: 'Fail', 1: 'Min. illuminance only', 2: 'Avg. illuminance only', 3: 'Pass'}

    metric_info_dict = {
        'pass_fail': {
            'type': 'VisualizationMetaData',
            'data_type': GenericType('Pass/Fail', '').to_dict(),
            'unit': '',
            'legend_parameters': pass_fail_lpar.to_dict()
        }
    }
    try:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        for metric, data in metric_info_dict.items():
            if not os.path.exists(os.path.join(output_folder, metric)):
                os.mkdir(os.path.join(output_folder, metric))
            file_path = os.path.join(output_folder, metric, 'vis_metadata.json')
            with open(file_path, 'w') as fp:
                json.dump(data, fp, indent=4)
    except Exception:
        _logger.exception('Failed to write the visualization metadata files.')
        sys.exit(1)
    else:
        sys.exit(0)
