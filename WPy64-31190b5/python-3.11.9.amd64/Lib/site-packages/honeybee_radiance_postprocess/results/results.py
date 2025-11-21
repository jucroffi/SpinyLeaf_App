"""Post-processing Results class."""
import json
from pathlib import Path
from itertools import islice, cycle
from typing import Tuple, Union, List
import itertools
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.generic import GenericType
from ladybug.datatype.base import DataTypeBase
from ladybug.dt import DateTime
from ladybug.header import Header

from ..annual import occupancy_schedule_8_to_6
from ..metrics import (average_values_array2d, cumulative_values_array2d,
    peak_values_array2d)
from ..util import filter_array, hoys_mask, check_array_dim, \
    _filter_grids_by_pattern
from .. import type_hints
from ..dynamic import DynamicSchedule, ApertureGroupSchedule

is_cpu = not is_gpu


class _ResultsFolder(object):
    """Base class for ResultsFolder.

    This class includes properties that are independent of the results.

    Args:
        folder: Path to results folder.

    Properties:
        * folder
        * grids_info
        * sun_up_hours
        * sun_down_hours
        * light_paths
        * default_states
        * grid_states
        * timestep
        * study_hours

    """
    __slots__ = ('_folder', '_grids_info', '_sun_up_hours', '_sun_down_hours',
                 '_sun_up_hours_mask', '_sun_down_hours_mask', '_datetimes',
                 '_light_paths', '_default_states', '_grid_states', '_timestep',
                 '_study_hours')

    def __init__(self, folder: Union[str, Path]):
        """Initialize ResultsFolder."""
        self._folder = Path(folder).absolute().as_posix()
        self._timestep, self._study_hours = self._get_study_info()
        self.grids_info = self._get_grids_info()
        self.sun_up_hours = self._get_sun_up_hours()
        self._sun_up_hours_mask = self._get_sun_up_hours_mask()
        self._sun_down_hours_mask = self._get_sun_down_hours_mask()
        self._datetimes = self._get_datetimes()
        self._light_paths = self._get_light_paths()
        self._default_states = self._get_default_states()
        self._grid_states = self._get_grid_states()

    @property
    def folder(self):
        """Return full path to results folder as a string."""
        return self._folder

    @property
    def grids_info(self):
        """Return grids information as list of dictionaries for each grid."""
        return self._grids_info

    @grids_info.setter
    def grids_info(self, grids_info):
        assert isinstance(grids_info, list), \
            f'Grids information must be a list. Got object of type: {type(grids_info)}.'
        for grid_info in grids_info:
            assert isinstance(grid_info, dict), \
                'Object in grids information must be a dictionary. ' \
                f'Got object of type {type(grid_info)}.'
            if not grid_info.get('light_path', None):
                # if light path key is nonexistent
                grid_info['light_path'] = [['__static_apertures__']]
        self._grids_info = grids_info

    @property
    def sun_up_hours(self):
        """Return sun up hours."""
        return self._sun_up_hours

    @sun_up_hours.setter
    def sun_up_hours(self, sun_up_hours):
        assert isinstance(sun_up_hours, list), \
            f'Sun up hours must be a list. Got object of type: {type(sun_up_hours)}'
        self._sun_up_hours = sun_up_hours
        self.sun_down_hours = np.setdiff1d(self.study_hours, sun_up_hours).tolist()

    @property
    def sun_up_hours_mask(self):
        """Return sun up hours masking array."""
        return self._sun_up_hours_mask

    @property
    def sun_down_hours(self):
        """Return sun down hours."""
        return self._sun_down_hours

    @sun_down_hours.setter
    def sun_down_hours(self, sun_down_hours):
        assert isinstance(sun_down_hours, list), \
            f'Sun down hours must be a list. Got object of type: {type(sun_down_hours)}'
        self._sun_down_hours = sun_down_hours

    @property
    def sun_down_hours_mask(self):
        """Return sun down hours masking array."""
        return self._sun_down_hours_mask

    @property
    def datetimes(self):
        """Return DateTimes for sun up hours."""
        return self._datetimes

    @property
    def light_paths(self):
        """Return the identifiers of the light paths."""
        return self._light_paths

    @property
    def default_states(self):
        """Return default states as a dictionary."""
        return self._default_states

    @property
    def grid_states(self):
        """Return grid states as a dictionary."""
        return self._grid_states

    @property
    def timestep(self):
        """Return timestep as an integer."""
        return self._timestep

    @property
    def study_hours(self):
        """Return study hours as a list."""
        return self._study_hours

    def _get_light_paths(self) -> list:
        """Find all light paths in grids_info."""
        lp = []
        for grid_info in self.grids_info:
            try:
                light_paths = grid_info['light_path']
            except KeyError:
                grid_info['light_path'] = [['__static_apertures__']]
                light_paths = grid_info['light_path']
            for light_path in light_paths:
                for elem in light_path:
                    if elem in lp:
                        continue
                    if elem == '__static_apertures__':
                        lp.insert(0, elem)
                    else:
                        lp.append(elem)
            if not light_paths and '__static_apertures__' not in lp:
                lp.insert(0, '__static_apertures__')

        return lp

    def _get_default_states(self) -> dict:
        """Set default state to 0 for all light paths."""
        default_states = {}
        for light_path in self.light_paths:
            default_states[light_path] = [0]
        return default_states

    def _get_grid_states(self) -> dict:
        """Read grid_states.json if available."""
        info = Path(self.folder, 'grid_states.json')
        if info.is_file():
            with open(info) as data_f:
                data = json.load(data_f)
            return data
        else:
            # only static results
            return {}

    def _get_study_info(self) -> Tuple[int, list]:
        """Read study info file."""
        study_info_file = Path(self.folder).joinpath('study_info.json')
        if study_info_file.exists():
            with open(study_info_file) as file:
                study_info = json.load(file)
            if study_info['timestep'] == 1:
                study_info['study_hours'] = \
                    list(map(int, study_info['study_hours']))
        else:
            study_info = {}
            study_info['timestep'] = 1
            study_info['study_hours'] = AnalysisPeriod().hoys

        return study_info['timestep'], study_info['study_hours']

    def _get_datetimes(self) -> List[DateTime]:
        """Get a list of DateTimes of the sun up hours."""
        datetimes = [
            DateTime.from_hoy(hoy) for hoy in list(map(float, self.sun_up_hours))
            ]

        return datetimes

    def _get_grids_info(self) -> List[dict]:
        """Get grids info from folder."""
        info = Path(self.folder, 'grids_info.json')
        with open(info) as data_f:
            grids = json.load(data_f)

        return grids

    def _get_sun_up_hours(self) -> List[float]:
        """Get sun up hours from folder."""
        suh_fp = Path(self.folder, 'sun-up-hours.txt')
        sun_up_hours = np.loadtxt(suh_fp, dtype=float).tolist()
        if self.timestep == 1:
            sun_up_hours = list(map(int, sun_up_hours))

        return sun_up_hours

    def _get_sun_up_hours_mask(self) -> List[int]:
        """Get a sun up hours masking array of the study hours."""
        sun_up_hours_mask = \
            np.where(np.isin(np.array(self.study_hours), np.array(self.sun_up_hours)))[0]

        return sun_up_hours_mask

    def _get_sun_down_hours_mask(self) -> List[int]:
        """Get a sun down hours masking array of the study hours."""
        sun_down_hours_mask = \
            np.where(~np.isin(np.array(self.study_hours), np.array(self.sun_up_hours)))[0]

        return sun_down_hours_mask

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.folder}'


class Results(_ResultsFolder):
    """Results class.

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
        * unit
        * cache_arrays
        * use_gpu
    """
    __slots__ = ('_schedule', '_occ_pattern', '_total_occ', '_sun_down_occ_hours',
                 '_occ_mask', '_arrays', '_valid_states', '_datatype', '_unit',
                 '_cache_arrays', '_use_gpu')

    def __init__(self, folder, datatype: DataTypeBase = None,
                 schedule: list = None, unit: str = None,
                 load_arrays: bool = False, cache_arrays: bool = True):
        """Initialize Results."""
        _ResultsFolder.__init__(self, folder)
        self.schedule = schedule
        self._arrays = self._load_arrays() if load_arrays else {}
        self._valid_states = self._get_valid_states()
        self.datatype = datatype
        self.unit = unit
        self.cache_arrays = cache_arrays

    @property
    def schedule(self):
        """Return schedule."""
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        self._schedule = schedule if schedule else \
            occupancy_schedule_8_to_6(timestep=self.timestep, as_list=True)
        self._update_occ()

    @property
    def occ_pattern(self):
        """Return a filtered version of the annual schedule that only includes the
        sun-up-hours."""
        return self._occ_pattern

    @property
    def total_occ(self):
        """Return an integer for the total occupied hours of the schedule."""
        return self._total_occ

    @property
    def sun_down_occ_hours(self):
        """Return an integer for the number of occupied hours where the sun is down and
        there's no possibility of being daylit or experiencing glare."""
        return self._sun_down_occ_hours

    @property
    def occ_mask(self):
        """Return an occupancy mask as a NumPy array that can be used to mask the
        results."""
        return self._occ_mask

    @property
    def arrays(self):
        """Return a dictionary of all the NumPy arrays that have been loaded."""
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self._arrays = value

    @property
    def valid_states(self):
        """Return a dictionary with valid states. Each light path is represented by a
        key-value pair where the light path identifier is the key and the value is a list
        of valid states, e.g., [0, 1, 2, ...]."""
        return self._valid_states

    @property
    def datatype(self):
        """Return a Ladybug DataType object."""
        return self._datatype

    @datatype.setter
    def datatype(self, value):
        if value is not None:
            assert isinstance(value, DataTypeBase), \
                f'data_type must be a Ladybug DataType. Got {type(value)}'
        else:
            value = GenericType('Generic', '')
        self._datatype = value

    @property
    def unit(self):
        """Return unit of hourly values."""
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value

    @property
    def cache_arrays(self):
        """Return a boolean to indicate whether arrays are cached."""
        return self._cache_arrays

    @cache_arrays.setter
    def cache_arrays(self, value):
        self._cache_arrays = value

    def total(
            self, hoys: list = [], states: DynamicSchedule = None,
            grids_filter: str = '*', res_type: str = 'total'
            ) -> type_hints.total:
        """Get summed values for each sensor.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with total values and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        mask = hoys_mask(self.sun_up_hours, hoys)

        total = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask
                )
                array_total = array_filter.sum(axis=1)
            else:
                array_total = np.zeros(grid_info['count'])
            total.append(array_total)

        return total, grids_info

    def point_in_time(
            self, datetime: Union[float, DateTime], states: DynamicSchedule = None,
            grids_filter: str = '*', res_type: str = 'total'
            ) -> type_hints.point_in_time:
        """Get point in time values.

        Args:
            datetime: Hour of the year as a float or DateTime object.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with point in time values and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        if isinstance(datetime, float):
            dt = DateTime.from_hoy(datetime)
        elif isinstance(datetime, DateTime):
            dt = datetime
        else:
            error_message = (
                f'Input datetime must be of type {int} or {DateTime}. '
                f'Received {type(DateTime)}.'
            )
            raise ValueError(error_message)

        idx = self._index_from_datetime(dt)

        pit_values = []
        for grid_info in grids_info:
            if idx:
                array = self._array_from_states(
                    grid_info, states=states, res_type=res_type)
                pit_values.append(array[:, idx])
            else:
                # datetime not in sun up hours, add zeros
                pit_values.append(np.zeros(grid_info['count']))

        return pit_values, grids_info

    def point_in_time_to_folder(
            self, target_folder: str, datetime: Union[float, DateTime],
            states: DynamicSchedule = None, grids_filter: str = '*',
            res_type: str = 'total'
            ) -> type_hints.point_in_time:
        """Get point in time values and write the values to a folder.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            datetime: Hour of the year as a float or DateTime object.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with point in time values and grid information.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        pit_values, grids_info = self.point_in_time(
            datetime=datetime, states=states,
            grids_filter=grids_filter, res_type=res_type)

        metric_folder = folder.joinpath('point_in_time')

        for count, grid_info in enumerate(grids_info):
            d = pit_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.pit')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

        return pit_values, grids_info

    def average_values(
            self, hoys: list = [], states: DynamicSchedule = None, grids_filter: str = '*',
            res_type: str = 'total') -> type_hints.average_values:
        """Get average values for each sensor over a given period.

        The hoys input can be used to filter the data for a particular time
        period.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with the average value for each sensor and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        full_length = len(self.study_hours) if len(hoys) == 0 else len(hoys)
        mask = hoys_mask(self.sun_up_hours, hoys)

        average_values = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                results = average_values_array2d(
                    array_filter, full_length)
            else:
                results = np.zeros(grid_info['count'])
            average_values.append(results)

        return average_values, grids_info

    def average_values_to_folder(
            self, target_folder: str, hoys: list = [], states: DynamicSchedule = None,
            grids_filter: str = '*', res_type: str = 'total'):
        """Get average values for each sensor over a given period and write the
        values to a folder.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        average_values, grids_info = self.average_values(
            hoys=hoys, states=states,
            grids_filter=grids_filter, res_type=res_type)

        metric_folder = folder.joinpath('average_values')

        for count, grid_info in enumerate(grids_info):
            d = average_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.average')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def median_values(
            self, hoys: list = [], states: dict = None, grids_filter: str = '*',
            res_type: str = 'total') -> type_hints.median_values:
        """Get median values for each sensor over a given period.

        The hoys input can be used to filter the data for a particular time
        period. If hoys is left empty the median values will likely be 0 since
        there are likely more sun down hours than sun up hours.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with the median value for each sensor and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        mask = hoys_mask(self.sun_up_hours, hoys)

        median_values = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                if not hoys:
                    # concatenate zero array
                    zero_array = \
                        np.zeros((grid_info['count'], len(self.sun_down_hours)))
                    array_filter = np.concatenate((array_filter, zero_array), axis=1)
                else:
                    # find number of hoys that are sun down hours
                    sdh_hoys = \
                        len(set(self.sun_down_hours).intersection(hoys))
                    if sdh_hoys != 0:
                        # concatenate zero array
                        zero_array = np.zeros((grid_info['count'], sdh_hoys))
                        array_filter = \
                            np.concatenate((array_filter, zero_array), axis=1)
                results = np.median(array_filter, axis=1)
            else:
                results = np.zeros(grid_info['count'])
            median_values.append(results)

        return median_values, grids_info

    def median_values_to_folder(
            self, target_folder: str, hoys: list = [], states: dict = None,
            grids_filter: str = '*', res_type: str = 'total'):
        """Get median values for each sensor over a given period and write the
        values to a folder.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        median_values, grids_info = self.median_values(
            hoys=hoys, states=states,
            grids_filter=grids_filter, res_type=res_type)

        metric_folder = folder.joinpath('median_values')

        for count, grid_info in enumerate(grids_info):
            d = median_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.median')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def cumulative_values(
            self, hoys: list = [], states: DynamicSchedule = None,
            t_step_multiplier: float = 1, grids_filter: str = '*',
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
        grids_info = self._filter_grids(grids_filter=grids_filter)
        mask = hoys_mask(self.sun_up_hours, hoys)

        cumulative_values = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                results = cumulative_values_array2d(
                    array_filter, self.timestep, t_step_multiplier)
            else:
                results = np.zeros(grid_info['count'])
            cumulative_values.append(results)

        return cumulative_values, grids_info

    def cumulative_values_to_folder(
            self, target_folder: str, hoys: list = [],
            states: DynamicSchedule = None, t_step_multiplier: float = 1,
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
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        cumulative_values, grids_info = self.cumulative_values(
            hoys=hoys, states=states, t_step_multiplier=t_step_multiplier,
            grids_filter=grids_filter, res_type=res_type
            )

        metric_folder = folder.joinpath('cumulative_values')

        for count, grid_info in enumerate(grids_info):
            d = cumulative_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.cumulative')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def peak_values(
            self, hoys: list = [], states: DynamicSchedule = None, grids_filter: str = '*',
            coincident: bool = False, res_type: str = 'total'
            ) -> type_hints.peak_values:
        """Get peak values for each sensor over a given period.

        The hoys input can be used to filter the data for a particular time
        period.

        Args:
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            coincident: Boolean to indicate whether output values represent the peak
                value for each sensor throughout the entire analysis (False) or they
                represent the highest overall value across each sensor grid at a
                particular timestep (True). Defaults to False.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with the peak value for each sensor and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        mask = hoys_mask(self.sun_up_hours, hoys)
        filt_suh = hoys if len(hoys) != 0 else self.sun_up_hours

        peak_values = []
        max_hoys = []
        for grid_info in grids_info:
            max_i = None
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            if np.any(array):
                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=mask)
                results, max_i = peak_values_array2d(
                    array_filter, coincident=coincident)
            else:
                results = np.zeros(grid_info['count'])
            peak_values.append(results)
            if max_i:
                max_hoys.append(filt_suh[max_i])
            else:
                max_hoys.append(None)

        return peak_values, max_hoys, grids_info

    def peak_values_to_folder(
            self, target_folder: str, hoys: list = [], states: DynamicSchedule = None,
            grids_filter: str = '*', coincident: bool = False, res_type='total'):
        """Get peak values for each sensor over a given period and write the
        values to a folder.

        Args:
            target_folder: Folder path to write peak values in. Usually this
                folder is called 'metrics'.
            hoys: An optional numbers or list of numbers to select the hours of
                the year (HOYs) for which results will be computed. Defaults to [].
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
            coincident: Boolean to indicate whether output values represent the peak
                value for each sensor throughout the entire analysis (False) or they
                represent the highest overall value across each sensor grid at a
                particular timestep (True). Defaults to False.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        peak_values, max_hoys, grids_info = self.peak_values(
            hoys=hoys, states=states, grids_filter=grids_filter,
            coincident=coincident, res_type=res_type)

        metric_folder = folder.joinpath('peak_values')

        for count, grid_info in enumerate(grids_info):
            d = peak_values[count]
            full_id = grid_info['full_id']
            output_file = metric_folder.joinpath(f'{full_id}.peak')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        max_hoys_file = metric_folder.joinpath('max_hoys.txt')
        max_hoys_file.write_text('\n'.join(str(h) for h in max_hoys))

        info_file = metric_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def _array_to_annual_data(
            self, grid_info, states: DynamicSchedule = None,
            sensor_index: list = None, res_type: str = 'total'
            ) -> Tuple[List[HourlyContinuousCollection], dict, list]:
        """Get annual data for one or multiple sensors.

        Args:
            grid_info: Grid information of the grid.
            states: A dictionary of states. Defaults to None.
            sensor_index: A list of sensor indices as integers. Defaults to None.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with Data Collections for each sensor, grid information,
                and a list of the sensors.
        """
        analysis_period = AnalysisPeriod(timestep=self.timestep)

        # if no sensor_index, create list with all sensors
        if not sensor_index:
            sensor_index = [range(grid_info['count'])]

        data_collections = []
        array = self._array_from_states(grid_info, states=states, res_type=res_type)
        for idx in sensor_index:
            if np.any(array):
                values = array[idx, :]
            else:
                values = np.zeros(len(self.sun_up_hours))
            annual_array = Results.values_to_annual(
                self.sun_up_hours, values, self.timestep, self.study_hours)
            header = Header(self.datatype, self.unit, analysis_period)
            header.metadata['sensor grid'] = grid_info['full_id']
            header.metadata['sensor index'] = idx
            data_collections.append(
                HourlyContinuousCollection(header, annual_array.tolist()))

        return data_collections, grid_info, sensor_index

    def annual_data(
            self, states: DynamicSchedule = None, grids_filter: str = '*',
            sensor_index: dict = None, res_type: str = 'total'
            ) -> type_hints.annual_data:
        """Get annual data for one or multiple sensors.

        Args:
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            sensor_index: A dictionary with grids as keys and a list of sensor
                indices as values. Defaults to None.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            Tuple: A tuple with Data Collections for each sensor, grid information,
                and a dictionary of the sensors.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        analysis_period = AnalysisPeriod(timestep=self.timestep)

        # if no sensor_index, create dict with all sensors
        if not sensor_index:
            sensor_index = {}
            for grid_info in grids_info:
                sensor_index[grid_info['full_id']] = \
                    [i for i in range(grid_info['count'])]

        data_collections = []
        for grid_info in grids_info:
            data_collections_grid = []
            grid_id = grid_info['full_id']
            array = self._array_from_states(grid_info, states=states, res_type=res_type)
            indices = sensor_index[grid_id]
            for idx in indices:
                if np.any(array):
                    values = array[idx, :]
                else:
                    values = np.zeros(len(self.sun_up_hours))
                annual_array = Results.values_to_annual(
                    self.sun_up_hours, values, self.timestep)
                header = Header(self.datatype, self.unit, analysis_period)
                header.metadata['sensor grid'] = grid_id
                header.metadata['sensor index'] = idx
                data_collections_grid.append(
                    HourlyContinuousCollection(header, annual_array.tolist()))
            data_collections.append(data_collections_grid)

        return data_collections, grids_info, sensor_index

    def annual_data_to_folder(
            self, target_folder: str, states: DynamicSchedule = None, grids_filter: str = '*',
            sensor_index: dict = None, res_type: str = 'total'):
        """Get annual data for one or multiple sensors and write the data to a
        folder as Data Collections.

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            sensor_index: A dictionary with grids as keys and a list of sensor
                indices as values. Defaults to None.
            res_type: Type of results to load. Defaults to 'total'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        data_collections, grids_info, sensor_index = self.annual_data(
            states=states, grids_filter=grids_filter, sensor_index=sensor_index,
            res_type=res_type)

        metric_folder = folder.joinpath('datacollections')

        for count, grid_info in enumerate(grids_info):
            grid_collections = data_collections[count]
            for data_collection in grid_collections:
                grid_id = grid_info['full_id']
                sensor_id = data_collection.header.metadata['sensor index']
                data_dict = data_collection.to_dict()
                data_file = metric_folder.joinpath(f'{grid_id}_{sensor_id}.json')
                data_file.parent.mkdir(parents=True, exist_ok=True)
                data_file.write_text(json.dumps(data_dict))

    @staticmethod
    def values_to_annual(
            hours: Union[List[float], np.ndarray],
            values: Union[List[float], np.ndarray],
            timestep: int, base_value: int = 0,
            dtype: np.dtype = np.float32) -> np.ndarray:
        """Map a 1D NumPy array based on a set of hours to an annual array.

        This method creates an array with a base value of length 8760 and
        replaces the base value with the input 'values' at the indices of the
        input 'hours'.

        Args:
            hours: A list of hours. This can be a regular list or a 1D NumPy
                array.
            values: A list of values to map to an annual array. This can be a
                regular list or a 1D NumPy array.
            timestep: Time step of the simulation.
            base_value: A value that will be applied for the base array.
            dtype: A NumPy dtype for the annual array.

        Returns:
            A 1D NumPy array.
        """
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if not isinstance(hours, np.ndarray):
            hours = np.array(hours)
        check_array_dim(values, 1)
        assert hours.shape == values.shape

        full_ap = np.array(AnalysisPeriod(timestep=timestep).hoys)
        indices = np.where(np.isin(full_ap, hours))[0]
        annual_array = np.repeat(np.array(base_value), 8760 * timestep).astype(dtype)

        annual_array[indices] = values

        return annual_array

    def _index_from_datetime(self, datetime: DateTime) -> Union[int, None]:
        """Returns the index of the input datetime in the list of datetimes
        from the datetimes property.

        If the DateTime is not in the list, the function will return None.

        Args:
            datetime: A DateTime object.

        Returns:
            Index as an integer or None.
        """
        assert isinstance(datetime, DateTime), \
            f'Expected DateTime object but received {type(datetime)}'
        try:
            index = self.datetimes.index(datetime)
        except Exception:
            # DateTime not in sun up hours
            index = None

        return index

    def _get_array(
            self, grid_info: dict, light_path: str, state: int = 0,
            res_type: str = 'total', extension: str = '.npy') -> np.ndarray:
        """Get an array for a given grid, light path, and state.

        The array will be fetched from the 'arrays' property if it has been
        loaded already.

        Args:
            grid_info: Grid information.
            light_path: Identifier of the light path.
            state: State as an integer. E.g., 0 for the default state.
                Defaults to 0.
            res_type: Type of results to load. Defaults to 'total'.
            extension: File extension of the array to load. Defaults to '.npy'.

        Returns:
            np.ndarray: A NumPy array of a given grid, light path, and state.
        """
        grid_id = grid_info['full_id']

        state_identifier = self._state_identifier(grid_id, light_path, state=state)
        try:
            array = self.arrays[grid_id][light_path][state_identifier][res_type]
        except Exception:
            array = self._load_array(
                grid_info, light_path, state=state, res_type=res_type,
                extension=extension
            )

        return array

    def _load_array(
            self, grid_info: dict, light_path: str, state: int = 0,
            res_type: str = 'total', extension: str = '.npy') -> np.ndarray:
        """Load a NumPy file to an array.

        This method will also update the arrays property value.

        Args:
            grid_info: Grid information.
            light_path: Identifier of the light path.
            state: State as an integer. E.g., 0 for the default state.
                Defaults to 0.
            res_type: Which type of result to return a file for. E.g., 'total' for total
                illuminance or 'direct' for direct illuminance.
            extension: File extension of the array to load. Defaults to '.npy'.

        Returns:
            np.ndarray: A NumPy array of a given grid, light path, and state
                from a NumPy file.
        """
        grid_id = grid_info['full_id']

        def merge_dicts(array_dict, arrays):
            for key, value in array_dict.items():
                if isinstance(value, dict):
                    node = arrays.setdefault(key, {})
                    merge_dicts(value, node)
                else:
                    arrays[key] = value
            return arrays

        state_identifier = self._state_identifier(grid_id, light_path, state=state)
        file = self._get_file(grid_id, light_path, state_identifier, res_type,
                              extension=extension)

        array = np.load(file)

        if self.cache_arrays:
            array_dict = {grid_id: {light_path: {state_identifier: {res_type: array}}}}
            arrays = merge_dicts(array_dict, self.arrays)
            self.arrays = arrays

        return array

    def clear_cached_arrays(self, res_type: str = None):
        """Clear the cached arrays.
        
        This method will simply clear the arrays property to remove arrays from
        memory, unless the res_type is selected in which case only 'total' or
        'direct' arrays will be deleted.
        
        Args:
            res_type: Which type of results to clear. This can be either
                'total' or 'direct'.
        """
        if res_type is not None:
            assert res_type in ('total', 'direct')
            arrays = self.arrays
            for grid_id in arrays:
                for light_path in arrays[grid_id]:
                    for state in arrays[grid_id][light_path]:
                        del arrays[grid_id][light_path][state][res_type]
        else:
            self.arrays.clear()

    def _state_identifier(
            self, grid_id: str, light_path: str, state: int = 0) -> Union[str, None]:
        """Get the state identifier from a light path and state integer.

        Args:
            grid_id: Grid identifier.
            light_path: Identifier of the light path.
            state: State as an integer. E.g., 0 for the default state.
                Defaults to 0.

        Returns:
            State identifier. For static apertures the identifier is 'default',
            and for other light paths it is the light path identifier preceded
            by the state integer, e.g., '0_light_path'. If the state is -1 the
            state identifier will be None.
        """
        # TODO: Figure out if there is a better way to handle the states.
        # I.e., state integer <--> state identifier.
        valid_states = self.valid_states[light_path]
        if state in valid_states:
            if light_path == '__static_apertures__':
                state_identifier = 'default'
            else:
                state_identifier = self.grid_states[grid_id][light_path][state]
            return state_identifier
        elif state == -1:
            return None
        else:
            error_message = (
                f'State of {light_path} must be any of {valid_states} for on '
                f'or -1 for off. Received state {state}.'
            )
            raise ValueError(error_message)

    def _get_file(
            self, grid_id: str, light_path: str, state_identifier: str,
            res_type: str = 'total', extension: str = '.npy') -> Path:
        """Return the path of a results file.

        Args:
            grid_id: Grid identifier.
            light_path: Identifier of the light path.
            state_identifier: State identifier.
            res_type: Which type of result to return a file for. E.g., 'total' for total
                illuminance or 'direct' for direct illuminance.
            extension: File extension of the array to load. Defaults to '.npy'.

        Returns:
            Path to a NumPy file.
        """
        file = Path(self.folder, light_path, state_identifier,
                    res_type, grid_id + extension)
        if not file.is_file():
            raise FileNotFoundError(f'File {file} not found in the results folder.')

        return file

    def _validate_dynamic_states(self, states: dict) -> dict:
        """Validate dynamic states and return states dictionary.

        If all light paths in the dictionary have 8760 values, the states
        dictionary is returned as is. If some light paths have less than 8760
        values, pattern of values will be repeated until it reaches a length of
        8760.

        Args:
            states: A dictionary of states.

        Returns:
            dict: A dictionary of states.
        """
        if all(len(values) == 8760 for values in states.values()):
            return states
        for light_path, values in states.items():
            if len(values) < 8760:
                states[light_path] = list(islice(cycle(values), 8760))
            elif len(values) > 8760:
                error_message = (
                    f'The light path {light_path} has {len(values)} values in '
                    f'its states schedule. Maximum allowed number of values '
                    f'is 8760.'
                )
                raise ValueError(error_message)

        return states

    def _validate_states(self, states: dict) -> dict:
        """Validate states and return states dictionary.

        If all light paths in the dictionary have integers only as values, the
        states dictionary is returned as is. If some light paths have values
        that are not integers, these values will be mapped as integers if
        possible, e.g., if the values are strings ('0', '1', etc.) instead of
        integers.

        Args:
            states: A dictionary of states.

        Returns:
            dict: A dictionary of states.
        """
        if all(isinstance(v, int) for values in states.values() for v in values):
            return states
        for light_path, values in states.items():
            try:
                states[light_path] = list(map(int, values))
            except ValueError as err:
                error_message = (
                    f'Failed to convert states schedule for light path '
                    f'{light_path} to integers.'
                )
                raise ValueError(error_message) from err

        return states

    def _filter_grid_states(self, grid_info, states: DynamicSchedule = None) -> DynamicSchedule:
        """Filter a dictionary of states by grid. Only light paths relevant to
        the given grid will be returned.

        Args:
            grid_info: Grid information.
            states: A dictionary of states. Light paths as keys and lists of
                8760 values for each key. The values should be integers
                matching the states or -1 for off. Default to None.

        Returns:
            dict: A filtered states dictionary.
        """
        light_paths = []
        for lp in grid_info['light_path']:
            for _lp in lp:
                if _lp == '__static_apertures__' and len(lp) > 1:
                    pass
                else:
                    light_paths.append(_lp)
        if states:
            states = states.filter_by_identifiers(light_paths)
        else:
            default_states = self.default_states
            states = DynamicSchedule()
            for light_path in light_paths:
                ap_group_schedule = ApertureGroupSchedule(
                    light_path, default_states[light_path], is_static=True)
                states.add_aperture_group_schedule(ap_group_schedule)

        return states

    def _array_from_states(
            self, grid_info, states: DynamicSchedule = None,
            res_type: str = 'total', zero_array: bool = False
            ) -> np.ndarray:
        """Create an array for a given grid by the states settings.

        Args:
            grid_info: Grid information of the grid.
            states: A dictionary of states. Light paths as keys and lists of 8760 values
                for each key. The values should be integers matching the states or -1 for
                off.
            res_type: Which type of result to create an array for. E.g., 'total'
                for total illuminance or 'direct' for direct illuminance.
            zero_array: Boolean to note if a 2D zero array should be created if
                the array of the grid is zero. This is the case if the
                illuminance of the grid is zero. (Default: False).

        Returns:
            A NumPy array based on the states settings.
        """
        # get states that are relevant for the grid
        states = self._filter_grid_states(grid_info, states=states)

        arrays = []
        for light_path, gr_schedule in states.dynamic_schedule.items():
            if gr_schedule.is_static:
                state = gr_schedule.schedule[0]
                # if state is -1 we continue since it is "turned off"
                if state == -1:
                    continue
                # load static array (state is static)
                array = self._get_array(
                    grid_info, light_path, state=state, res_type=res_type)
                arrays.append(array)
            else:
                # slice states to match sun up hours
                states_array = np.array(gr_schedule.schedule)[
                    np.array(self.sun_up_hours, int)]

                unique_states = np.unique(states_array)
                unique_states = unique_states[unique_states != -1]  # skip -1
                temp_arrays = []
                for state in unique_states:
                    state = int(state)
                    # load static array (state is static)
                    _array = self._get_array(
                        grid_info, light_path, state=state, res_type=res_type)
                    # get indices and add values to base array
                    state_indices = (states_array == state)
                    masked_array = np.zeros_like(_array)
                    masked_array[:, state_indices] = _array[:, state_indices]
                    temp_arrays.append(masked_array)
                if temp_arrays:
                    array = np.sum(np.stack(temp_arrays), axis=0)
                else:
                    array = np.zeros((grid_info['count'], len(self.sun_up_hours)))
                arrays.append(array)

        if len(arrays) == 0:
            array = np.zeros((grid_info['count'], len(self.sun_up_hours)))
        else:
            array = np.sum(np.stack(arrays, axis=0), axis=0)

        if not np.any(array):
            if not zero_array:
                array = np.asarray([])

        return array

    def _update_occ(self):
        """Set properties related to occupancy."""
        occ_mask = np.array(self.schedule, dtype=int)[self.sun_up_hours_mask]
        sun_down_sch = \
            np.array(self.schedule, dtype=int)[self.sun_down_hours_mask].sum()

        self._occ_mask = occ_mask
        self._total_occ = sum(self.schedule)
        self._sun_down_occ_hours = sun_down_sch

    def _filter_grids(self, grids_filter: str = '*') -> list:
        """Return grids information.

        Args:
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            list: List of grid information for filtered grids.
        """
        if grids_filter != '*':
            grids_info = \
                _filter_grids_by_pattern(self.grids_info, grids_filter)
        else:
            grids_info = self.grids_info

        return grids_info

    def _load_arrays(self) -> dict:
        """Load all the NumPy arrays in the results folder."""
        arrays = {}
        grids_info = self.grids_info

        for grid_info in grids_info:
            grid_id = grid_info['full_id']
            light_paths = grid_info['light_path']
            arrays[grid_id] = {}
            for light_path in light_paths:
                light_path = light_path[0]
                arrays[grid_id][light_path] = {}
                light_path_folder = Path(self.folder, light_path)
                for state_folder in Path(light_path_folder).iterdir():
                    state = state_folder.name
                    arrays[grid_id][light_path][state] = {}
                    for res_type_folder in Path(state_folder).iterdir():
                        res_type = res_type_folder.name
                        file = Path(res_type_folder, grid_id + '.npy')
                        array = np.load(file)
                        arrays[grid_id][light_path][state][res_type] = array

        return arrays

    def _get_valid_states(self) -> dict:
        """Returns a dictionary with valid states for each light path.

        For each light path there will be a key (identifier of the light path)
        and its value will be a list of valid states as integers.

        Example of output format:
        {
            '__static_apertures__': [0],
            'Room1_North': [0, 1],
            'Room1_South': [0, 1],
            'Room2_North1': [0, 1],
            'Room2_North2': [0, 1]
        }

        Returns:
            dict: Valid states integers for each light path.
        """
        valid_states = {}
        grid_states = self.grid_states
        if '__static_apertures__' in self.light_paths:
            valid_states['__static_apertures__'] = [0]
        for light_paths in grid_states.values():
            for light_path, states in light_paths.items():
                if light_path not in valid_states:
                    valid_states[light_path] = list(range(len(states)))

        return valid_states

    def _light_paths_from_grid_info(self, grid_info: Union[dict, str]) -> list:
        if isinstance(grid_info, str):
            for _grid_info in self.grids_info:
                if _grid_info['full_id'] == grid_info:
                    grid_info = _grid_info
                    break
            else:
                raise Exception(f'Grid info with full_id "{grid_info}" not found.')
        light_paths = []
        for lp in grid_info['light_path']:
            for _lp in lp:
                if _lp == '__static_apertures__' and len(lp) > 1:
                    pass
                else:
                    light_paths.append(_lp)

        return light_paths

    def _get_state_combinations(self, grid_info: Union[dict, str]) -> List[dict]:
        light_paths = self._light_paths_from_grid_info(grid_info)
        valid_states = self._get_valid_states()
        filtered_states = {lp: valid_states[lp] for lp in light_paths}
        keys, values = zip(*filtered_states.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        return combinations
