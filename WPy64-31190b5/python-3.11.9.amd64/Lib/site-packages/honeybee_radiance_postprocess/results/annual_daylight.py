import json
from pathlib import Path
from typing import Tuple, List
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from collections import defaultdict

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.illuminance import Illuminance
from ladybug.datatype.fraction import Fraction
from ladybug.header import Header

from ..annual import occupancy_schedule_8_to_6
from ..metrics import da_array2d, cda_array2d, udi_array2d, udi_lower_array2d, \
    udi_upper_array2d, ase_array2d
from ..util import filter_array, filter_array2d
from ..annualdaylight import _annual_daylight_vis_metadata
from ..electriclight import array_to_dimming_fraction
from .. import type_hints
from ..dynamic import DynamicSchedule, ApertureGroupSchedule
from .results import Results

is_cpu = not is_gpu


class AnnualDaylight(Results):
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
        Results.__init__(self, folder, datatype=Illuminance('Illuminance'),
                         schedule=schedule, unit='lux', load_arrays=load_arrays,
                         cache_arrays=cache_arrays)

    def daylight_autonomy(
            self, threshold: float = 300, states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate daylight autonomy.

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the daylight autonomy and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        da = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = filter_array2d(array, mask=self.occ_mask)
                results = da_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
            else:
                results = np.zeros(grid_info['count'])
            da.append(results)

        return da, grids_info

    def continuous_daylight_autonomy(
            self, threshold: float = 300, states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate continuous daylight autonomy.

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the continuous daylight autonomy and grid
                information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        cda = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = filter_array2d(array, mask=self.occ_mask)
                results = cda_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
            else:
                results = np.zeros(grid_info['count'])
            cda.append(results)

        return cda, grids_info

    def useful_daylight_illuminance(
            self, min_t: float = 100, max_t: float = 3000, states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate useful daylight illuminance.

        Args:
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the useful daylight illuminance and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        udi = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = filter_array2d(array, mask=self.occ_mask)
                results = udi_array2d(
                    array_filter, total_occ=self.total_occ, min_t=min_t, max_t=max_t)
            else:
                results = np.zeros(grid_info['count'])
            udi.append(results)

        return udi, grids_info

    def useful_daylight_illuminance_lower(
            self, min_t: float = 100, states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate lower than useful daylight illuminance.

        Args:
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the lower than useful daylight illuminance and
                grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_down_occ_hours = self.sun_down_occ_hours

        udi_lower = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = filter_array2d(array, mask=self.occ_mask)
                results = udi_lower_array2d(
                    array_filter, total_occ=self.total_occ,
                    min_t=min_t, sun_down_occ_hours=sun_down_occ_hours)
            else:
                results = np.zeros(grid_info['count'])
            udi_lower.append(results)

        return udi_lower, grids_info

    def useful_daylight_illuminance_upper(
            self, max_t: float = 3000, states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_metric:
        """Calculate higher than useful daylight illuminance.

        Args:
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the higher than useful daylight illuminance and
                grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        udi_upper = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = filter_array2d(array, mask=self.occ_mask)
                results = udi_upper_array2d(
                    array_filter, total_occ=self.total_occ, max_t=max_t)
            else:
                results = np.zeros(grid_info['count'])
            udi_upper.append(results)

        return udi_upper, grids_info

    def annual_metrics(
            self, threshold: float = 300, min_t: float = 100,
            max_t: float = 3000, states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_daylight_metrics:
        """Calculate multiple annual daylight metrics.

        This method will calculate the following metrics:
            * Daylight autonomy
            * Continuous daylight autonomy
            * Useful daylight illuminance
            * Lower than useful daylight illuminance
            * Higher than useful daylight illuminance

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the five annual daylight metrics and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_down_occ_hours = self.sun_down_occ_hours

        da = []
        cda = []
        udi = []
        udi_lower = []
        udi_upper = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                array_filter = filter_array2d(array, mask=self.occ_mask)
                da_results = da_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
                cda_results = cda_array2d(
                    array_filter, total_occ=self.total_occ, threshold=threshold)
                udi_results = udi_array2d(
                    array_filter, total_occ=self.total_occ, min_t=min_t, max_t=max_t)
                udi_lower_results = udi_lower_array2d(
                    array_filter, total_occ=self.total_occ, min_t=min_t,
                    sun_down_occ_hours=sun_down_occ_hours)
                udi_upper_results = udi_upper_array2d(
                    array_filter, total_occ=self.total_occ, max_t=max_t)
            else:
                da_results = cda_results = udi_results = udi_lower_results = \
                    udi_upper_results = np.zeros(grid_info['count'])
            da.append(da_results)
            cda.append(cda_results)
            udi.append(udi_results)
            udi_lower.append(udi_lower_results)
            udi_upper.append(udi_upper_results)

        return da, cda, udi, udi_lower, udi_upper, grids_info

    def annual_metrics_to_folder(
            self, target_folder: str, threshold: float = 300,
            min_t: float = 100, max_t: float = 3000, states: DynamicSchedule = None,
            grids_filter: str = '*'):
        """Calculate and write multiple annual daylight metrics to a folder.

        This method will calculate the following metrics:
            * Daylight autonomy
            * Continuous daylight autonomy
            * Useful daylight illuminance
            * Lower than useful daylight illuminance
            * Higher than useful daylight illuminance

        Args:
            target_folder: Folder path to write annual metrics in. Usually this
                folder is called 'metrics'.
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            min_t: Minimum threshold for useful daylight illuminance. Defaults to 100.
            max_t: Maximum threshold for useful daylight illuminance. Defaults to 3000.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        da, cda, udi, udi_lower, udi_upper, grids_info = self.annual_metrics(
            threshold=threshold, min_t=min_t, max_t=max_t, states=states,
            grids_filter=grids_filter)

        pattern = {
            'da': da, 'cda': cda, 'udi_lower': udi_lower, 'udi': udi,
            'udi_upper': udi_upper
        }
        for metric, data in pattern.items():
            metric_folder = folder.joinpath(metric)
            extension = metric.split('_')[0]
            for count, grid_info in enumerate(grids_info):
                d = data[count]
                full_id = grid_info['full_id']
                output_file = metric_folder.joinpath(f'{full_id}.{extension}')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(output_file, d, fmt='%.2f')

        for metric in pattern.keys():
            info_file = folder.joinpath(metric, 'grids_info.json')
            info_file.write_text(json.dumps(grids_info))

        metric_info_dict = _annual_daylight_vis_metadata()
        for metric, data in metric_info_dict.items():
            vis_metadata_file = folder.joinpath(metric, 'vis_metadata.json')
            vis_metadata_file.write_text(json.dumps(data, indent=4))

    def spatial_daylight_autonomy(
            self, threshold: float = 300, target_time: float = 50,
            states: DynamicSchedule = None, grids_filter: str = '*'
            ) -> type_hints.spatial_daylight_autonomy:
        """Calculate spatial daylight autonomy.

        Note: This component will only output a LEED compliant sDA if you've
        run the simulation with blinds and blinds schedules as per the
        IES-LM-83-12. Use the states option to calculate a LEED compliant sDA.

        Args:
            threshold: Threshold value for daylight autonomy. Defaults to 300.
            target_time: A minimum threshold of occupied time (eg. 50% of the
                time), above which a given sensor passes and contributes to the
                spatial daylight autonomy. Defaults to 50.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the spatial daylight autonomy and grid
                information.
        """
        da, grids_info = self.daylight_autonomy(
           threshold=threshold, states=states, grids_filter=grids_filter)

        sda = []
        for array in da:
            sda.append((array >= target_time).mean())

        return sda, grids_info

    def annual_sunlight_exposure(
            self, direct_threshold: float = 1000, occ_hours: int = 250,
            states: DynamicSchedule = None, grids_filter: str = '*'
            ) -> type_hints.annual_sunlight_exposure:
        """Calculate annual sunlight exposure.

        Args:
            direct_threshold: The threshold that determines if a sensor is
                overlit. Defaults to 1000.
            occ_hours: The number of occupied hours that cannot receive more
                than the direct_threshold. Defaults to 250.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the annual sunlight exposure, the number of
                hours that exceeds the direct threshold for each sensor, and
                grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        ase = []
        hours_above = []
        for grid_info in grids_info:
            array = self._array_from_states(
                grid_info, states=states, res_type='direct')
            if np.any(array):
                array_filter = filter_array2d(array, mask=self.occ_mask)
                results, h_above = ase_array2d(
                    array_filter, occ_hours=occ_hours,
                    direct_threshold=direct_threshold)
            else:
                results = np.float64(0)
                h_above = np.zeros(grid_info['count'])
            ase.append(results)
            hours_above.append(h_above)

        return ase, hours_above, grids_info

    def annual_sunlight_exposure_to_folder(
            self, target_folder: str, direct_threshold: float = 1000,
            occ_hours: int = 250, states: DynamicSchedule = None,
            grids_filter: str = '*'):
        """Calculate and write annual sunlight exposure to a folder.

        Args:
            direct_threshold: The threshold that determines if a sensor is
                overlit. Defaults to 1000.
            occ_hours: The number of occupied hours that cannot receive more
                than the direct_threshold. Defaults to 250.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        ase, hours_above, grids_info = self.annual_sunlight_exposure(
            direct_threshold=direct_threshold, occ_hours=occ_hours,
            states=states, grids_filter=grids_filter
            )

        pattern = {'ase': ase, 'hours_above': hours_above}
        for metric, data in pattern.items():
            metric_folder = folder.joinpath(metric)
            if metric == 'hours_above':
                extension = 'res'
            else:
                extension = 'ase'
            for count, grid_info in enumerate(grids_info):
                d = data[count]
                full_id = grid_info['full_id']
                output_file = metric_folder.joinpath(f'{full_id}.{extension}')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                if metric == 'hours_above':
                    np.savetxt(output_file, d, fmt='%i')
                elif metric == 'ase':
                    output_file.write_text('%.2f' % d)

        for metric in pattern.keys():
            info_file = folder.joinpath(metric, 'grids_info.json')
            info_file.write_text(json.dumps(grids_info))

    def daylight_control_schedules(
            self, states: DynamicSchedule = None, grids_filter: str = '*',
            base_schedule: list = None, ill_setpoint: float = 300,
            min_power_in: float = 0.3, min_light_out: float = 0.2,
            off_at_min: bool = False
            ) -> Tuple[List[np.ndarray], List[dict]]:
        """Generate electric lighting schedules from annual daylight results.

        Such controls will dim the lights according to whether the illuminance values
        at the sensor locations are at a target illuminance setpoint. The results can be
        used to account for daylight controls in energy simulations.

        This function will generate one schedule per sensor grid in the simulation. Each
        grid should have sensors at the locations in space where daylight dimming sensors
        are located. Grids with one, two, or more sensors can be used to model setups
        where fractions of each room are controlled by different sensors. If the sensor
        grids are distributed over the entire floor of the rooms, the resulting schedules
        will be idealized, where light dimming has been optimized to supply the minimum
        illuminance setpoint everywhere in the room.

        Args:
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            base_schedule: A list of 8760 fractional values for the lighting schedule
                representing the usage of lights without any daylight controls. The
                values of this schedule will be multiplied by the hourly dimming
                fraction to yield the output lighting schedules. If None, a schedule
                from 9AM to 5PM on weekdays will be used. (Default: None).
            ill_setpoint: A number for the illuminance setpoint in lux beyond which
                electric lights are dimmed if there is sufficient daylight.
                Some common setpoints are listed below. (Default: 300 lux).

                * 50 lux - Corridors and hallways.
                * 150 lux - Computer work spaces (screens provide illumination).
                * 300 lux - Paper work spaces (reading from surfaces that need illumination).
                * 500 lux - Retail spaces or museums illuminating merchandise/artifacts.
                * 1000 lux - Operating rooms and workshops where light is needed for safety.

            min_power_in: A number between 0 and 1 for the the lowest power the lighting
                system can dim down to, expressed as a fraction of maximum
                input power. (Default: 0.3).
            min_light_out: A number between 0 and 1 the lowest lighting output the lighting
                system can dim down to, expressed as a fraction of maximum light
                output. Note that setting this to 1 means lights aren't dimmed at
                all until the illuminance setpoint is reached. This can be used to
                approximate manual light-switching behavior when used in conjunction
                with the off_at_min input below. (Default: 0.2).
            off_at_min: Boolean to note whether lights should switch off completely when
                they get to the minimum power input. (Default: False).

        Returns:
            A tuple with two values.

            -   schedules: A list of lists where each sub-list represents an electric
                lighting dimming schedule for a sensor grid.

            -   grids_info: A list of grid information.
        """
        # process the base schedule input into a list of values
        if base_schedule is None:
            base_schedule = occupancy_schedule_8_to_6(timestep=self.timestep)
        base_schedule = np.array(base_schedule)

        grids_info = self._filter_grids(grids_filter=grids_filter)
        sun_up_hours = [int(h) for h in self.sun_up_hours]

        dim_fracts = []
        for grid_info in grids_info:
            array = self._array_from_states(
                grid_info, states=states, res_type='total')
            if np.any(array):
                fract_list = array_to_dimming_fraction(
                    array, sun_up_hours, ill_setpoint, min_power_in,
                    min_light_out, off_at_min
                )
            else:
                fract_list = np.ones(8760)
            dim_fracts.append(fract_list)

        schedules = []
        for grid_info, dim_fract in zip(grids_info, dim_fracts):
            sch_vals = base_schedule * dim_fract
            schedules.append(sch_vals)

        return schedules, grids_info

    def daylight_control_schedules_to_folder(
            self, target_folder: str, states: DynamicSchedule = None,
            grids_filter: str = '*', base_schedule: list = None,
            ill_setpoint: float = 300, min_power_in: float = 0.3,
            min_light_out: float = 0.2, off_at_min: bool = False):
        """Generate electric lighting schedules from annual daylight results and
        write the schedules to a folder.

        Such controls will dim the lights according to whether the illuminance values
        at the sensor locations are at a target illuminance setpoint. The results can be
        used to account for daylight controls in energy simulations.

        This function will generate one schedule per sensor grid in the simulation. Each
        grid should have sensors at the locations in space where daylight dimming sensors
        are located. Grids with one, two, or more sensors can be used to model setups
        where fractions of each room are controlled by different sensors. If the sensor
        grids are distributed over the entire floor of the rooms, the resulting schedules
        will be idealized, where light dimming has been optimized to supply the minimum
        illuminance setpoint everywhere in the room.

        Args:
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            base_schedule: A list of 8760 fractional values for the lighting schedule
                representing the usage of lights without any daylight controls. The
                values of this schedule will be multiplied by the hourly dimming
                fraction to yield the output lighting schedules. If None, a schedule
                from 9AM to 5PM on weekdays will be used. (Default: None).
            ill_setpoint: A number for the illuminance setpoint in lux beyond which
                electric lights are dimmed if there is sufficient daylight.
                Some common setpoints are listed below. (Default: 300 lux).

                * 50 lux - Corridors and hallways.
                * 150 lux - Computer work spaces (screens provide illumination).
                * 300 lux - Paper work spaces (reading from surfaces that need illumination).
                * 500 lux - Retail spaces or museums illuminating merchandise/artifacts.
                * 1000 lux - Operating rooms and workshops where light is needed for safety.

            min_power_in: A number between 0 and 1 for the the lowest power the lighting
                system can dim down to, expressed as a fraction of maximum
                input power. (Default: 0.3).
            min_light_out: A number between 0 and 1 the lowest lighting output the lighting
                system can dim down to, expressed as a fraction of maximum light
                output. Note that setting this to 1 means lights aren't dimmed at
                all until the illuminance setpoint is reached. This can be used to
                approximate manual light-switching behavior when used in conjunction
                with the off_at_min input below. (Default: 0.2).
            off_at_min: Boolean to note whether lights should switch off completely when
                they get to the minimum power input. (Default: False).
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        schedules, grids_info = self.daylight_control_schedules(
            states=states, grids_filter=grids_filter,
            base_schedule=base_schedule, ill_setpoint=ill_setpoint,
            min_power_in=min_power_in, min_light_out=min_light_out,
            off_at_min=off_at_min)

        schedule_folder = folder.joinpath('control_schedules')

        for count, grid_info in enumerate(grids_info):
            d = schedules[count]
            full_id = grid_info['full_id']
            output_file = schedule_folder.joinpath(f'{full_id}.txt')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_file, d, fmt='%.2f')

        info_file = schedule_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def annual_uniformity_ratio(
            self, threshold: float = 0.5, states: DynamicSchedule = None,
            grids_filter: str = '*') -> type_hints.annual_uniformity_ratio:
        """Calculate annual uniformity ratio.

        Args:
            threshold: A threshold for the uniformity ratio. Defaults to 0.5.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the annual uniformity ratio, annual
                data collections, and grid information.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)
        analysis_period = AnalysisPeriod(timestep=self.timestep)

        data_collections = []
        annual_uniformity_ratio = []
        for grid_info in grids_info:
            array = self._array_from_states(grid_info, states=states, res_type='total')
            if np.any(array):
                su_min_array = array.min(axis=0)
                su_mean_array = array.mean(axis=0)
                su_uniformity_ratio = su_min_array / su_mean_array

                array_filter = np.apply_along_axis(
                    filter_array, 1, array, mask=self.occ_mask)
                min_array = array_filter.min(axis=0)
                mean_array = array_filter.mean(axis=0)
                uniformity_ratio = min_array / mean_array
                annual_uniformity_ratio.append(
                    np.float64(
                        (uniformity_ratio >= threshold).sum() / self.total_occ * 100
                    )
                )
            else:
                su_uniformity_ratio = np.zeros(len(self.sun_up_hours))
                annual_uniformity_ratio.append(np.float64(0))

            annual_array = \
                self.values_to_annual(
                    self.sun_up_hours, su_uniformity_ratio, self.timestep)
            header = Header(Fraction(), '%', analysis_period)
            header.metadata['sensor grid'] = grid_info['full_id']
            data_collections.append(
                HourlyContinuousCollection(header, annual_array.tolist()))

        return annual_uniformity_ratio, data_collections, grids_info

    def annual_uniformity_ratio_to_folder(
            self, target_folder: str, threshold: float = 0.5,
            states: DynamicSchedule = None, grids_filter: str = '*'
            ):
        """Calculate annual uniformity ratio and write it to a folder.

        Args:
            target_folder: Folder path to write annual uniformity ratio in.
            threshold: A threshold for the uniformity ratio. Defaults to 0.5.
            states: A dictionary of states. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.

        Returns:
            Tuple: A tuple with the daylight autonomy and grid information.
        """
        folder = Path(target_folder)
        folder.mkdir(parents=True, exist_ok=True)

        annual_uniformity_ratio, data_collections, grids_info = \
            self.annual_uniformity_ratio(threshold=threshold, states=states,
                                         grids_filter=grids_filter)

        datacollection_folder = folder.joinpath('datacollections')
        uniformity_ratio_folder = folder.joinpath('uniformity_ratio')

        for aur, data_collection, grid_info in \
            zip(annual_uniformity_ratio, data_collections, grids_info):
            grid_id = grid_info['full_id']
            data_dict = data_collection.to_dict()
            data_file = datacollection_folder.joinpath(f'{grid_id}.json')
            data_file.parent.mkdir(parents=True, exist_ok=True)
            data_file.write_text(json.dumps(data_dict))

            aur_file = uniformity_ratio_folder.joinpath(f'{grid_id}.ur')
            aur_file.parent.mkdir(parents=True, exist_ok=True)
            aur_file.write_text(str(round(aur, 2)))

        info_file = uniformity_ratio_folder.joinpath('grids_info.json')
        info_file.write_text(json.dumps(grids_info))

    def dynamic_schedule_from_sensor_maximum(
            self, sensor_index: dict, grids_filter: str = '*',
            maximum: float = 3000, res_type: str = 'total') -> DynamicSchedule:
        """Calculate a DynamicSchedule from a sensor and a maximum allowed
        illuminance.

        Args:
            sensor_index: A dictionary with grids as keys and a list of sensor
                indices as values. Defaults to None.
            grids_filter: The name of a grid or a pattern to filter the grids.
                Defaults to '*'.
            maximum: A float value of the maximum illuminance allowed for the
                sensor.
            res_type: Type of results to load. Defaults to 'total'.

        Returns:
            DynamicSchedule object.
        """
        grids_info = self._filter_grids(grids_filter=grids_filter)

        aperture_group_schedules = []
        for grid_info in grids_info:
            control_sensor = sensor_index.get(grid_info['full_id'], None)
            if control_sensor is None:
                continue
            assert len(control_sensor) == 1, ('Expected one control sensor for '
                f'grid {grid_info["name"]}. Received {len(control_sensor)} '
                'control sensors.')
            control_sensor_index = control_sensor[0]

            combinations = self._get_state_combinations(grid_info)

            array_list_combinations = []
            for combination in combinations:
                combination_arrays = []
                for light_path, state_index in combination.items():
                    array = self._get_array(
                        grid_info, light_path, state=state_index, res_type=res_type)
                    sensor_array = array[control_sensor_index,:]
                    combination_arrays.append(sensor_array)
                combination_array = sum(combination_arrays)
                array_list_combinations.append(combination_array)
            array_combinations = np.array(array_list_combinations)
            array_combinations[array_combinations > maximum] = -np.inf
            max_indices = array_combinations.argmax(axis=0)
            combinations = [combinations[idx] for idx in max_indices]

            states_schedule = defaultdict(list)
            for combination in combinations:
                for light_path, state_index in combination.items():
                    if light_path != '__static_apertures__':
                        states_schedule[light_path].append(state_index)

            # map states to 8760 values
            for light_path, state_indices in states_schedule.items():
                mapped_states = self.values_to_annual(
                    self.sun_up_hours, state_indices, self.timestep)
                mapped_states = mapped_states.astype(int)
                aperture_group_schedules.append(
                    ApertureGroupSchedule(light_path, mapped_states.tolist()))

        dyn_sch = DynamicSchedule.from_group_schedules(aperture_group_schedules)

        return dyn_sch
