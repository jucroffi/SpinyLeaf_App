"""Functions for EN 17037 post-processing."""
from typing import Union
from pathlib import Path
import json
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ladybug.color import Colorset
from ladybug.datatype.fraction import Fraction
from ladybug.legend import LegendParameters

from .results.annual_daylight import AnnualDaylight
from .dynamic import DynamicSchedule
from .metrics import da_array2d
from .util import filter_array


def en17037_to_files(
        array: np.ndarray, metrics_folder: Path, grid_info: dict) -> list:
    """Compute annual EN 17037 metrics for a NumPy array and write the results
    to a folder.

    This function generates 6 different files for daylight autonomy based on
    the varying level of recommendation in EN 17037.

    Args:
        array: A 2D NumPy array.
        metrics_folder: An output folder where the results will be written to.
            The folder will be created if it does not exist.
        grid_info: A grid information dictionary.

    Returns:
        tuple -- Tuple of lists of paths for da, sda, and compliance folders.
    """
    recommendations = {
        'minimum_illuminance': {
            'minimum': 100,
            'medium': 300,
            'high': 500
        },
        'target_illuminance': {
            'minimum': 300,
            'medium': 500,
            'high': 750
        }
    }
    compliance_value = {
        'minimum': 1,
        'medium': 2,
        'high': 3
    }

    grid_id = grid_info['full_id']
    grid_count = grid_info['count']

    da_folders = []
    sda_folders = []
    compliance_folders = []
    da_folder = metrics_folder.joinpath('da')
    sda_folder = metrics_folder.joinpath('sda')
    compliance_folder = metrics_folder.joinpath('compliance_level')

    for target_type, thresholds in recommendations.items():
        compliance_level = None
        for level, threshold in thresholds.items():
            # da
            da_level_folder = \
                da_folder.joinpath('_'.join([target_type, str(threshold)]))
            da_file = da_level_folder.joinpath(f'{grid_id}.da')
            if not da_file.parent.is_dir():
                da_file.parent.mkdir(parents=True)
            da = da_array2d(array, total_occ=4380, threshold=threshold)
            np.savetxt(da_file, da, fmt='%.2f')

            # sda
            sda_level_folder = \
                sda_folder.joinpath('_'.join([target_type, str(threshold)]))
            sda_file = sda_level_folder.joinpath(f'{grid_id}.sda')
            if not sda_file.parent.is_dir():
                sda_file.parent.mkdir(parents=True)
            sda = (da >= 50).mean() * 100
            with open(sda_file, 'w') as sdaf:
                sdaf.write(str(round(sda, 2)))

            space_target = 50 if target_type == 'target_illuminance' else 95
            if sda >= space_target:
                compliance_level = np.full((grid_count), compliance_value[level], dtype=int)

            da_folders.append(da_file.parent)
            sda_folders.append(sda_file.parent)

        if compliance_level is None:
            compliance_level = np.zeros(grid_count, dtype=int)
        compliance_level_folder = compliance_folder.joinpath(target_type)
        compliance_level_file = compliance_level_folder.joinpath(f'{grid_id}.pf')
        if not compliance_level_file.parent.is_dir():
            compliance_level_file.parent.mkdir(parents=True)
        np.savetxt(compliance_level_file, compliance_level, fmt='%i')
        compliance_folders.append(compliance_level_file.parent)

    return da_folders, sda_folders, compliance_folders


def en17037_to_folder(
        results: Union[str, AnnualDaylight], schedule: list,
        states: DynamicSchedule = None, grids_filter: str = '*',
        sub_folder: str = 'en17037') -> Path:
    """Compute annual EN 17037 metrics in a folder and write them in a subfolder.

    The results is an output folder of annual daylight recipe.

    Args:
        results: Results folder.
        schedule: An annual schedule for 8760 hours of the year as a list of
            values. This should be a daylight hours schedule.
        grids_filter: A pattern to filter the grids. By default all the grids
            will be processed.
        states: A dictionary of states. Defaults to None.
        sub_folder: An optional relative path for subfolder to copy results
            files. Default: en17037.

    Returns:
        str -- Path to results folder.
    """
    if not isinstance(results, AnnualDaylight):
        results = AnnualDaylight(results, schedule=schedule)
    else:
        results.schedule = schedule

    total_occ = results.total_occ
    occ_mask = results.occ_mask

    grids_info = results._filter_grids(grids_filter=grids_filter)

    sub_folder = Path(sub_folder)

    if total_occ != 4380:
        raise ValueError(
            f'There are {total_occ} occupied hours in the schedule. According '
            'to EN 17037 the schedule must consist of the daylight hours '
            'which is defined as the half of the year with the largest '
            'quantity of daylight')

    for grid_info in grids_info:
        array = results._array_from_states(
            grid_info, states=states, res_type='total', zero_array=True)
        if np.any(array):
            array = np.apply_along_axis(
                filter_array, 1, array, occ_mask)
        da_folders, sda_folders, compliance_folders = en17037_to_files(
            array, sub_folder, grid_info)

    # copy grids_info.json to all results folders
    for folder in da_folders + sda_folders + compliance_folders:
        grids_info_file = Path(folder, 'grids_info.json')
        with open(grids_info_file, 'w') as outf:
            json.dump(grids_info, outf, indent=2)

    metric_info_dict = _annual_daylight_en17037_vis_metadata()
    da_folder = sub_folder.joinpath('da')
    for metric, data in metric_info_dict.items():
        file_path = da_folder.joinpath(metric, 'vis_metadata.json')
        with open(file_path, 'w') as fp:
            json.dump(data, fp, indent=4)

    return sub_folder


def _annual_daylight_en17037_vis_metadata():
    """Return visualization metadata for annual daylight."""
    da_lpar = LegendParameters(min=0, max=100, colors=Colorset.annual_comfort())

    metric_info_dict = {
        'minimum_illuminance_100': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 100 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'minimum_illuminance_300': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 300 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'minimum_illuminance_500': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - minimum 500 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'target_illuminance_300': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 300 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'target_illuminance_500': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 500 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        },
        'target_illuminance_750': {
            'type': 'VisualizationMetaData',
            'data_type': Fraction('Daylight Autonomy - target 750 lux').to_dict(),
            'unit': '%',
            'legend_parameters': da_lpar.to_dict()
        }
    }

    return metric_info_dict


def _annual_daylight_en17037_config():
    """Return vtk-config for annual daylight EN 17037. """
    cfg = {
        "data": [
            {
                "identifier": "Daylight Autonomy - target 300 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "target_illuminance/minimum/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - target 500 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "target_illuminance/medium/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - target 750 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "target_illuminance/high/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - minimum 100 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "minimum_illuminance/minimum/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - minimum 300 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "minimum_illuminance/medium/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
            {
                "identifier": "Daylight Autonomy - minimum 500 lux",
                "object_type": "grid",
                "unit": "Percentage",
                "path": "minimum_illuminance/high/da",
                "hide": False,
                "legend_parameters": {
                    "hide_legend": False,
                    "min": 0,
                    "max": 100,
                    "color_set": "nuanced",
                },
            },
        ]
    }

    return cfg
