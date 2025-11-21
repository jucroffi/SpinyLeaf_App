"""Functions for IES LM post-processing."""
from typing import Tuple, Union
from collections import defaultdict
import itertools
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from honeybee_radiance.postprocess.annual import filter_schedule_by_hours

from ..annual import schedule_to_hoys, occupancy_schedule_8_to_6
from ..results.annual_daylight import AnnualDaylight
from ..util import filter_array
from ..dynamic import DynamicSchedule, ApertureGroupSchedule
from .lm_schedule import shd_trans_schedule_descending, states_schedule_descending


def shade_transmittance_per_light_path(
    light_paths: list, shade_transmittance: Union[float, dict],
    shd_trans_dict: dict) -> dict:
    """Filter shade_transmittance by light paths and add default multiplier.

    Args:
        light_paths: A list of light paths.
        shade_transmittance: A value to use as a multiplier in place of solar
            shading. This input can be either a single value that will be used
            for all aperture groups, or a dictionary where aperture groups are
            keys, and the value for each key is the shade transmittance. Values
            for shade transmittance must be 1 > value > 0.
        shd_trans_dict: A dictionary used to store shade transmittance value
            for each aperture group.

    Returns:
        A dictionary with filtered light paths.
    """
    shade_transmittances = {}
    if isinstance(shade_transmittance, dict):
        for light_path in light_paths:
            # default multiplier
            shade_transmittances[light_path] = [1]
            # add custom shade transmittance
            if light_path in shade_transmittance:
                shade_transmittances[light_path].append(
                    shade_transmittance[light_path])
                shd_trans_dict[light_path] = shade_transmittance[light_path]
            # add default shade transmittance (0.02)
            elif light_path != '__static_apertures__':
                shade_transmittances[light_path].append(0.02)
                shd_trans_dict[light_path] = 0.02
            else:
                shade_transmittances[light_path].append(1)
                shd_trans_dict[light_path] = 1
    else:
        shd_trans = float(shade_transmittance)
        for light_path in light_paths:
            # default multiplier
            shade_transmittances[light_path] = [1]
            # add custom shade transmittance
            if light_path != '__static_apertures__':
                shade_transmittances[light_path].append(shd_trans)
                shd_trans_dict[light_path] = shd_trans
            else:
                shade_transmittances[light_path].append(1)
                shd_trans_dict[light_path] = 1

    return shade_transmittances, shd_trans_dict


def dynamic_schedule_direct_illuminance(
        results: Union[str, AnnualDaylight], grids_filter: str = '*',
        shade_transmittance: Union[float, dict] = 0.02,
        use_states: bool = False
        ) -> Tuple[dict, dict]:
    """Calculate a schedule of each aperture group.

    This function calculates an annual shading schedule of each aperture
    group. Hour by hour it will select the least shaded aperture group
    configuration, so that no more than 2% of the sensors points receive
    direct illuminance of 1000 lux or more.

    Args:
        results: Path to results folder or a Results class object.
        grids_filter: The name of a grid or a pattern to filter the grids.
            Defaults to '*'.
        shade_transmittance: A value to use as a multiplier in place of solar
            shading. This input can be either a single value that will be used
            for all aperture groups, or a dictionary where aperture groups are
            keys, and the value for each key is the shade transmittance. Values
            for shade transmittance must be 1 > value > 0.
            Defaults to 0.02.
        use_states: A boolean to note whether to use the simulated states. Set
            to True to use the simulated states. The default is False which will
            use the shade transmittance instead.

    Returns:
        Tuple: A tuple with a dictionary of the annual schedule and a
            dictionary of hours where no shading configuration comply with the
            2% rule.
    """
    if not isinstance(results, AnnualDaylight):
        results = AnnualDaylight(results)

    grids_info = results._filter_grids(grids_filter=grids_filter)
    schedule = occupancy_schedule_8_to_6(as_list=True)
    occ_pattern = \
        filter_schedule_by_hours(results.sun_up_hours, schedule=schedule)[0]
    occ_mask = np.array(occ_pattern)

    states_schedule = defaultdict(list)
    fail_to_comply = {}
    shd_trans_dict = {}

    for grid_info in grids_info:
        grid_states_schedule = defaultdict(list)

        grid_count = grid_info['count']
        light_paths = []
        for lp in grid_info['light_path']:
            for _lp in lp:
                if _lp == '__static_apertures__' and len(lp) > 1:
                    pass
                else:
                    light_paths.append(_lp)

        shade_transmittances, shd_trans_dict = (
            shade_transmittance_per_light_path(
                light_paths, shade_transmittance, shd_trans_dict
            )
        )

        if len(light_paths) > 6:
            if use_states:
                grid_states_schedule, fail_to_comply = states_schedule_descending(
                    results, grid_info, light_paths, occ_mask,
                    grid_states_schedule, fail_to_comply)
            else:
                grid_states_schedule, fail_to_comply = shd_trans_schedule_descending(
                    results, grid_info, light_paths, shade_transmittances, occ_mask,
                    grid_states_schedule, fail_to_comply)
        else:
            if use_states:
                combinations = results._get_state_combinations(grid_info)
            else:
                shade_transmittances, shd_trans_dict = shade_transmittance_per_light_path(
                    light_paths, shade_transmittance, shd_trans_dict)
                keys, values = zip(*shade_transmittances.items())
                combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            array_list_combinations = []
            for combination in combinations:
                combination_arrays = []
                for light_path, value in combination.items():
                    if use_states:
                        combination_arrays.append(
                            results._get_array(grid_info, light_path, state=value,
                                               res_type='direct')
                        )
                    else:
                        array = results._get_array(
                            grid_info, light_path, res_type='direct')
                        if value == 1:
                            combination_arrays.append(array)
                        else:
                            combination_arrays.append(array * value)
                combination_array = sum(combination_arrays)

                combination_percentage = \
                    (combination_array >= 1000).sum(axis=0) / grid_count
                array_list_combinations.append(combination_percentage)
            array_combinations = np.array(array_list_combinations)
            array_combinations[array_combinations > 0.02] = -np.inf

            grid_comply = np.where(np.all(array_combinations==-np.inf, axis=0))[0]
            if grid_comply.size != 0:
                grid_comply = np.array(results.sun_up_hours)[grid_comply]
                fail_to_comply[grid_info['name']] = \
                    [int(hoy) for hoy in grid_comply]

            array_combinations_filter = np.apply_along_axis(
                filter_array, 1, array_combinations, occ_mask
            )
            max_indices = array_combinations_filter.argmax(axis=0)
            # select the combination for each hour
            combinations = [combinations[idx] for idx in max_indices]

            # merge the combinations of dicts
            for combination in combinations:
                for light_path, value in combination.items():
                    if light_path != '__static_apertures__':
                        grid_states_schedule[light_path].append(value)

        for key, value in grid_states_schedule.items():
            if key not in states_schedule:
                states_schedule[key] = value
            else:
                if use_states:
                    merged_array = np.logical_or(states_schedule[key], value).astype(int)
                else:
                    merged_array = np.minimum(states_schedule[key], value)
                states_schedule[key] = merged_array

    occupancy_hoys = schedule_to_hoys(schedule, results.sun_up_hours)

    # map states to 8760 values
    if use_states:
        aperture_group_schedules = []
        for identifier, values in states_schedule.items():
            mapped_states = results.values_to_annual(
                occupancy_hoys, values, results.timestep, dtype=np.int32)
            aperture_group_schedules.append(
                ApertureGroupSchedule(identifier, mapped_states.tolist())
            )
        states_schedule = \
            DynamicSchedule.from_group_schedules(aperture_group_schedules)
    else:
        for light_path, shd_trans in states_schedule.items():
            mapped_states = results.values_to_annual(
                occupancy_hoys, shd_trans, results.timestep)
            states_schedule[light_path] = mapped_states

    return states_schedule, fail_to_comply, shd_trans_dict
