"""honeybee-radiance-postprocess WELL commands."""
import sys
import logging
import json
import os
import click

from ladybug.color import Color
from ladybug.datatype.generic import GenericType
from ladybug.legend import LegendParameters

from ..well.well import well_annual_daylight

_logger = logging.getLogger(__name__)


@click.group(help='Commands for WELL post-processing of Radiance results.')
def well():
    pass


@well.command('well-annual-daylight')
@click.argument(
    'folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)
)
@click.argument(
    'daylight-hours',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--grids-filter', '-gf', help='A pattern to filter the grids.', default='*',
    show_default=True
)
@click.option(
    '--sub-folder', '-sf', help='Relative path for subfolder to write output '
    'files.', default='well_summary', show_default=True
)
def well_daylight(
    folder, daylight_hours, grids_filter, sub_folder
):
    """Calculate credits for WELL L06.

    Use the shade-transmittance option to set a shade transmittance values for
    aperture groups. The shade-transmittance-file option takes precedence over
    the shade-transmittance, however, if any aperture groups are missing in the
    JSON file given to the shade-transmittance-file option, the value from
    shade-transmittance will be used for those aperture groups.

    \b
    Args:
        folder: Results folder. This folder is an output folder of annual daylight
            recipe. The daylight simulation must include aperture groups.
        daylight_hours: Daylight hours schedule for EN 17037.
    """
    with open(daylight_hours) as hourly_schedule:
        daylight_hours = [int(float(v)) for v in hourly_schedule]

    try:
        well_annual_daylight(
            folder, daylight_hours, grids_filter=grids_filter, sub_folder=sub_folder
        )
    except Exception:
        _logger.exception('Failed to generate WELL summary.')
        sys.exit(1)
    else:
        sys.exit(0)


@well.command('well-daylight-vis-metadata')
@click.option(
    '--output-folder', '-o', help='Output folder for vis metadata files.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    default='visualization', show_default=True
)
def well_daylight_vis(output_folder):
    """Write visualization metadata files for WELL Daylight."""
    colors = [Color(220, 0, 0), Color(0, 220, 0)]
    pass_fail_lpar = \
        LegendParameters(min=0, max=1, colors=colors, segment_count=2, title='Pass/Fail')
    pass_fail_lpar.ordinal_dictionary = {0: "Fail", 1: "Pass"}

    metric_info_dict = {
        'L01': {
            'type': 'VisualizationMetaData',
            'data_type': GenericType('sDA200,40%', '').to_dict(),
            'unit': '',
            'legend_parameters': pass_fail_lpar.to_dict()
        },
        'L06': {
            'type': 'VisualizationMetaData',
            'data_type': GenericType('sDA300,50%', '').to_dict(),
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
