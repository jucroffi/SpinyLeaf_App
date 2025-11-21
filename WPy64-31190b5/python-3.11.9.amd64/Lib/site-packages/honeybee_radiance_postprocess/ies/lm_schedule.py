"""Module for dynamic LM schedules."""
from typing import Tuple
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ..results.annual_daylight import AnnualDaylight
from ..util import filter_array


def shd_trans_schedule_descending(
        results: AnnualDaylight, grid_info, light_paths, shade_transmittances, occ_mask,
        states_schedule, fail_to_comply
        ) -> Tuple[dict, dict]:
    grid_count = grid_info['count']
    full_direct = []
    full_thresh = []
    full_shd_trans_array = []
    for light_path in light_paths:
        array = results._get_array(grid_info, light_path, res_type="direct")
        array = np.apply_along_axis(filter_array, 1, array, occ_mask)
        full_direct.append(array)
        full_thresh.append((array >= 1000).sum(axis=0))
        full_shd_trans_array.append(shade_transmittances[light_path][1])

    # Sum the array element-wise.
    # This array is the sum of all direct illuminance without shade
    # transmittance.
    full_direct_sum = sum(full_direct)

    # Create base list of shading combinations (all set to 1).
    # We will replace the 1s later.
    combinations = [
        {light_path: 1 for light_path in light_paths}
        for i in range(full_direct_sum.shape[1])
    ]

    # Find the percentage of floor area >= 1000 lux.
    # This array is the percentage for each hour (axis=0).
    direct_pct_above = (full_direct_sum >= 1000).sum(axis=0) / grid_count

    # Find the indices where the percentage of floor area is > 2%.
    # This array is the problematic hours.
    above_2_indices = np.where(direct_pct_above > 0.02)[0]

    # Use the indices to get the relevant hours.
    direct_sum = np.take(full_direct_sum, above_2_indices, axis=1)

    # Use the indices to get the relevant hours.
    direct = np.take(full_direct, above_2_indices, axis=2)

    # Use the indices to get the relevant hours.
    thresh = np.take(full_thresh, above_2_indices, axis=1)

    # Sort and get indices. Negate the array to get descending order.
    # Descending order puts the "highest offender" light path first.
    sort_thresh = np.argsort(-thresh, axis=0).transpose()

    _combinations = []
    _combinations.insert(
        0, (np.arange(full_direct_sum.shape[1]), combinations)
    )

    if np.any(above_2_indices):
        # There are hours where the percentage of floor area is > 2%.
        for idx, lp in enumerate(light_paths):
            # Take column. For each iteration it will take the next column
            # in descending order, i.e., the "highest offender" is the first
            # column.
            sort_indices = np.take(sort_thresh, idx, axis=1)

            # Map light path identifiers to indices.
            light_path_ids = np.take(light_paths, sort_indices)

            # Map shade transmittance to indices.
            shd_trans_array = np.take(full_shd_trans_array, sort_indices)

            # Create combination for the subset.
            _subset_combination = [
                {light_path: _shd_trans} for light_path, _shd_trans in
                zip(light_path_ids, shd_trans_array)
            ]
            _combinations.insert(0, (above_2_indices, _subset_combination))

            # Take the values from each array by indexing.
            direct_array = \
                direct[sort_indices, :, range(len(sort_indices))].transpose()

            # Subtract the illuminance values.
            direct_sum = direct_sum - (direct_array * (1 - shd_trans_array))

            # Find the percentage of floor area >= 1000 lux.
            direct_pct_above = (direct_sum >= 1000).sum(axis=0) / grid_count

            # Find the indices where the percentage of floor area is > 2%.
            above_2_indices = np.where(direct_pct_above > 0.02)[0]

            # Break if there are no hours above 2%.
            if not np.any(above_2_indices):
                break

            # Update variables for the next iteration.
            direct_sum = np.take(direct_sum, above_2_indices, axis=1)
            direct = np.take(direct, above_2_indices, axis=2)
            thresh = np.take(thresh, above_2_indices, axis=1)
            sort_thresh = np.take(sort_thresh, above_2_indices, axis=0)

        if np.any(above_2_indices):
            # There are hours not complying with the 2% rule.
            previous_indices = []
            previous_combination = []
            grid_comply = []
            # Merge the combinations from the iterations of the subsets.
            for i, subset in enumerate(_combinations):
                if i == 0:
                    previous_indices = subset[0]
                else:
                    _indices = subset[0]
                    grid_comply = []
                    for _pr_idx in previous_indices:
                        grid_comply.append(_indices[_pr_idx])
                    previous_indices = grid_comply
            # Convert indices to sun up hours indices.
            filter_indices = np.where(occ_mask.astype(bool))[0]
            grid_comply = [filter_indices[_gc] for _gc in grid_comply]
            grid_comply = np.array(results.sun_up_hours)[grid_comply]
            fail_to_comply[grid_info['name']] = \
                [int(hoy) for hoy in grid_comply]

        previous_indices = None
        previous_combination = None
        # Merge the combinations from the iterations of the subsets.
        for i, subset in enumerate(_combinations):
            if i == 0:
                previous_indices, previous_combination = subset
            else:
                _indices, _combination = subset
                for _pr_idx, _pr_comb in \
                        zip(previous_indices, previous_combination):
                    for light_path, _shd_trans in _pr_comb.items():
                        _combination[_pr_idx][light_path] = _shd_trans
                previous_indices = _indices
                previous_combination = _combination

        combinations = _combination

    # Merge the combinations of dicts.
    for combination in combinations:
        for light_path, shd_trans in combination.items():
            if light_path != "__static_apertures__":
                states_schedule[light_path].append(shd_trans)

    return states_schedule, fail_to_comply


def states_schedule_descending(
        results: AnnualDaylight, grid_info, light_paths, occ_mask,
        states_schedule, fail_to_comply
        ) -> Tuple[dict, dict]:
    grid_count = grid_info['count']
    full_direct = []
    full_thresh = []
    full_direct_blinds = []
    for light_path in light_paths:
        array = results._get_array(
            grid_info, light_path, state=0, res_type="direct")
        array = np.apply_along_axis(filter_array, 1, array, occ_mask)
        full_direct.append(array)
        full_thresh.append((array >= 1000).sum(axis=0))

        array = results._get_array(
            grid_info, light_path, state=1, res_type="direct")
        array = np.apply_along_axis(filter_array, 1, array, occ_mask)
        full_direct_blinds.append(array)

    full_direct = np.array(full_direct)
    full_direct_blinds = np.array(full_direct_blinds)
    full_direct_sum = full_direct.sum(axis=0)

    new_array = full_direct.copy()

    percentage_sensors = (full_direct_sum >= 1000).sum(axis=0) / grid_count
    if not np.any(percentage_sensors > 0.02):
        combinations = [
            {light_path: 0 for light_path in light_paths}
            for i in range(full_direct_sum.shape[1])]
    else:
        tracking_array = np.zeros(
            (new_array.shape[0], new_array.shape[2]), dtype=int)

        percentage_sensors = (full_direct >= 1000).sum(axis=1) / grid_count

        ranking_indices = np.argsort(-percentage_sensors, axis=0)

        for rank in range(ranking_indices.shape[0]):
            # Calculate the percentage of sensors with values >= 1000 for the current new_array
            summed_array = np.sum(new_array, axis=0)
            percentage_sensors_summed = np.sum(
                summed_array >= 1000, axis=0) / grid_count
            indices_above_2_percent = np.where(
                percentage_sensors_summed > 0.02)[0]

            # Exit if there are no more hours exceeding the threshold
            if len(indices_above_2_percent) == 0:
                break

            # Array indices to use for replacement for these hours
            replace_indices = indices_above_2_percent
            array_indices = ranking_indices[rank, replace_indices]

            # Use advanced indexing to replace values in new_array for these hours
            for hour_idx, array_idx in zip(replace_indices, array_indices):
                new_array[array_idx, :, hour_idx] = full_direct_blinds[
                    array_idx, :, hour_idx
                ]

            # Update the tracking array
            tracking_array[array_indices, replace_indices] = 1

        combinations = []
        for hour in range(new_array.shape[2]):
            hour_dict = {
                light_paths[i]: tracking_array[i, hour]
                for i in range(tracking_array.shape[0])}
            combinations.append(hour_dict)

        final_summed_array = np.sum(new_array, axis=0)
        final_percentage_sensors_summed = (
            final_summed_array >= 1000).sum(
            axis=0) / grid_count
        final_indices_above_2_percent = np.where(
            final_percentage_sensors_summed > 0.02)[0]
        if np.any(final_indices_above_2_percent):
            sun_up_hours_indices = np.where(occ_mask == 1)[0][
                final_indices_above_2_percent]
            grid_comply = np.array(results.sun_up_hours)[sun_up_hours_indices]
            fail_to_comply[grid_info['name']] = [
                int(hoy) for hoy in grid_comply]

    for combination in combinations:
        for light_path, value in combination.items():
            if light_path != '__static_apertures__':
                states_schedule[light_path].append(value)

    return states_schedule, fail_to_comply
