import json
from pathlib import Path
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ladybug.datatype.energyflux import EnergyFlux

from ..metrics import (average_values_array2d, cumulative_values_array2d,
    peak_values_array2d)
from ..util import filter_array, hoys_mask
from ..annualirradiance import _annual_irradiance_vis_metadata
from .. import type_hints
from ..dynamic import DynamicSchedule
from .results import Results


class AnnualIrradiance(Results):
    """Annual Daylight Results class.

    Args:
        folder: Path to results folder.
        schedule: 8760 values as a list. Values must be either 0 or 1. Values of 1
            indicates occupied hours. If no schedule is provided a default schedule
            will be used. (Default: None).
        load_arrays: Set to True to load all NumPy arrays. If False the arrays will be
            loaded only once they are needed. In both cases the loaded array(s) will be
            stored in a dictionary under the arrays property. (Default: False).

    Properties:
        * schedule
        * occ_pattern
        * total_occ
        * sun_down_occ_hours
        * occ_mask
        * arrays
        * valid_states
        * datatype
    """
    def __init__(self, folder, schedule: list = None, load_arrays: bool = False,
                 cache_arrays: bool = False):
        """Initialize Results."""
        Results.__init__(self, folder, datatype=EnergyFlux('Irradiance'),
                         schedule=schedule, unit='W/m2', load_arrays=load_arrays,
                         cache_arrays=cache_arrays)

    def cumulative_values(
            self, hoys: list = [], states: DynamicSchedule = None,
            t_step_multiplier: float = 1000, grids_filter: str = '*',
            res_type: str = 'total') -> type_hints.cumulative_values:
        """Get cumulative values for each sensor over a given period.

        The hoys input can be used to filter the data for a particular time
        period.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            t_step_multiplier: A value that will be multiplied with the timestep.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with the cumulative value for each sensor and grid
                information.
        """
        cumulative_values, grids_info = \
            super(AnnualIrradiance, self).cumulative_values(
                hoys=hoys, states=states, t_step_multiplier=t_step_multiplier,
                grids_filter=grids_filter, res_type=res_type
                )

        return cumulative_values, grids_info

    def cumulative_values_to_folder(
            self, target_folder: str, hoys: list = [],
            states: DynamicSchedule = None, t_step_multiplier: float = 1000,
            grids_filter: str = '*', res_type: str = 'total'):
        """Get cumulative values for each sensor over a given period and write
        the values to a folder.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            t_step_multiplier: A value that will be multiplied with the timestep.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.
        """
        super(AnnualIrradiance, self).cumulative_values_to_folder(
            target_folder, hoys=hoys, states=states,
            t_step_multiplier=t_step_multiplier, grids_filter=grids_filter,
            res_type=res_type
            )

    def annual_metrics(
            self, hoys: list = [], states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_irradiance_metrics:
        """Calculate multiple annual irradiance metrics.

        This method will calculate the following metrics:
            * Average Irradiance (W/m2)
            * Peak Irradiance (W/m2)
            * Cumulative Radiation (kWh/m2)

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the three annual irradiance metrics and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        mask = hoys_mask(self.sun_up_hours, hoys)
        full_length = len(self.study_hours)

        average = []
        peak = []
        cumulative = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                average_results = average_values_array2d(
                    array_filter, full_length=full_length)
                peak_results, max_i = peak_values_array2d(
                    array_filter)
                cumulative_results = cumulative_values_array2d(
                    array_filter, self.timestep, 1000)
            else:
                average_results = peak_results = cumulative_results = \
                    np.zeros(grid_info['count'])
            average.append(average_results)
            peak.append(peak_results)
            cumulative.append(cumulative_results)

        return average, peak, cumulative, grids_info

    def annual_metrics_to_folder(
            self, target_folder: str, hoys: list = [],
            states: DynamicSchedule = None, grids_filter: str = '*'):
        """Calculate and write multiple annual irradiance metrics to a folder.

        This command generates 3 files for each input grid.
            * average_irradiance/{grid-name}.res -- Average Irradiance (W/m2)
            * peak_irradiance/{grid-name}.res -- Peak Irradiance (W/m2)
            * cumulative_radiation/{grid-name}.res -- Cumulative Radiation (kWh/m2)

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        average, peak, cumulative, grids_info = self.annual_metrics(
            hoys=hoys, states=states, grids_filter=grids_filter)

        pattern = {
            'average_irradiance': average, 'peak_irradiance': peak,
            'cumulative_radiation': cumulative,
        }
        for metric, data in pattern.items():
            metric_folder = folder.joinpath(metric)
            for count, grid_info in enumerate(grids_info):
                d = data[count]
                full_id = grid_info['full_id']
                output_file = metric_folder.joinpath(f'{full_id}.res')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(output_file, d, fmt='%.2f')

        for metric in pattern.keys():
            info_file = folder.joinpath(metric, 'grids_info.json')
            info_file.write_text(json.dumps(grids_info))

        metric_info_dict = _annual_irradiance_vis_metadata()
        for metric, data in metric_info_dict.items():
            vis_metadata_file = folder.joinpath(metric, 'vis_metadata.json')
            vis_metadata_file.write_text(json.dumps(data, indent=4))
