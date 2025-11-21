"""Functions for WELL post-processing."""
from typing import Tuple, Union
from pathlib import Path
import json
import numpy as np

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.generic import GenericType
from ladybug.header import Header
from honeybee.model import Model
from honeybee.units import conversion_factor_to_meters
from honeybee_radiance.writer import _filter_by_pattern

from ..metrics import da_array2d
from ..annual import occupancy_schedule_8_to_6
from ..results.annual_daylight import AnnualDaylight
from ..util import filter_array, recursive_dict_merge
from ..ies.lm import dynamic_schedule_direct_illuminance
from ..en17037 import en17037_to_folder


def _create_grid_summary(
    grid_info, sda_grid, sda_blinds_up_grid, sda_blinds_down_grid, pass_sda,
    total_floor, area_weighted=True):
    """Create a WELL summary for a single grid.

    Args:
        grid_info: Grid information.
        sda_grid: Spatial Daylight Autonomy.
        pass_sda: The percentage of the sensor points or floor area that
            passes sDA.
        total_floor: The number of sensor points or floor area.
        area_weighted: Boolean to determine if the results are area
            weighted. Defaults to True.

    Returns:
        Tuple:
        -   summary_grid: Summary of each grid individually.
    """
    grid_id = grid_info['full_id']
    grid_name = grid_info['name']
    grid_summary = {
        grid_id: {}
    }

    if area_weighted:
        _grid_summary = {
            grid_id: {
                'name': grid_name,
                'full_id': grid_id,
                'sda': round(sda_grid, 2),
                'sda_blinds_up': round(sda_blinds_up_grid, 2),
                'sda_blinds_down': round(sda_blinds_down_grid, 2),
                'floor_area_passing_sda': round(pass_sda, 2),
                'total_floor_area': round(total_floor, 2)
            }
        }
    else:
        _grid_summary = {
            grid_id: {
                'name': grid_name,
                'full_id': grid_id,
                'sda': round(sda_grid, 2),
                'sda_blinds_up': round(sda_blinds_up_grid, 2),
                'sda_blinds_down': round(sda_blinds_down_grid, 2),
                'sensor_count_passing_sda': int(round(pass_sda, 2)),
                'total_sensor_count': total_floor
            }
        }

    recursive_dict_merge(grid_summary, _grid_summary)

    return grid_summary


def _well_summary(
    pass_sda_grids: list, grids_info: list,
    grid_areas: list, pass_sda_blinds_up_grids: list,
    pass_sda_blinds_down_grids: list) -> Tuple[dict, dict]:
    """Create combined summary and summary for each grid individually.

    Args:
        pass_sda_grids: A list where each sublist is a list of True/False that
            tells if each sensor point passes sDA.
        grids_info: A list of grid information.
        grid_areas: A list where each sublist is the area of each sensor point.
            The alternative is a list of None values for each grid information.

    Returns:
        Tuple:
        -   summary: Summary of of all grids combined.
        -   summary_grid: Summary of each grid individually.
    """
    summary = {}
    summary_grid = {}

    if all(grid_area is not None for grid_area in grid_areas):
        # weighted by mesh face area
        total_area = 0
        total_area_pass_sda = 0
        for (pass_sda, grid_area, grid_info, pass_sda_blinds_up,
             pass_sda_blinds_down) in \
            zip(pass_sda_grids, grid_areas, grids_info,
                pass_sda_blinds_up_grids, pass_sda_blinds_down_grids):
            total_grid_area = grid_area.sum()

            area_pass_sda = grid_area[pass_sda].sum()
            area_pass_sda_blind_up = grid_area[pass_sda_blinds_up].sum()
            area_pass_sda_blinds_down = grid_area[pass_sda_blinds_down].sum()
            sda_grid = area_pass_sda / total_grid_area * 100
            sda_blinds_up_grid = area_pass_sda_blind_up / total_grid_area * 100
            sda_blinds_down_grid = area_pass_sda_blinds_down / total_grid_area * 100

            # grid summary
            grid_summary = \
                _create_grid_summary(
                    grid_info, sda_grid, sda_blinds_up_grid, sda_blinds_down_grid,
                    area_pass_sda, total_grid_area, area_weighted=True
                )

            recursive_dict_merge(summary_grid, grid_summary)

            total_area += total_grid_area
            total_area_pass_sda += area_pass_sda

        summary['sda'] = round(total_area_pass_sda / total_area * 100, 2)
        summary['floor_area_passing_sda'] = total_area_pass_sda
        summary['total_floor_area'] = total_area
    else:
        # assume all sensor points cover the same area
        total_sensor_count = 0
        total_sensor_count_pass_sda = 0
        for (pass_sda, grid_info, pass_sda_blinds_up, pass_sda_blinds_down) in \
            zip(pass_sda_grids, grids_info, pass_sda_blinds_up_grids,
                pass_sda_blinds_down_grids):
            grid_count = grid_info['count']

            sensor_count_pass_sda = pass_sda.sum()
            sensor_count_pass_sda_blinds_up = pass_sda_blinds_up.sum()
            sensor_count_pass_sda_blinds_down = pass_sda_blinds_down.sum()
            sda_grid = sensor_count_pass_sda / grid_count * 100
            sda_blinds_up_grid = sensor_count_pass_sda_blinds_up / grid_count * 100
            sda_blinds_down_grid = sensor_count_pass_sda_blinds_down / grid_count * 100

            # grid summary
            grid_summary = \
                _create_grid_summary(
                    grid_info, sda_grid, sda_blinds_up_grid, sda_blinds_down_grid,
                    sensor_count_pass_sda, grid_count, area_weighted=False
                )

            recursive_dict_merge(summary_grid, grid_summary)

            total_sensor_count += grid_count
            total_sensor_count_pass_sda += sensor_count_pass_sda

        summary['sda'] = round(total_sensor_count_pass_sda / total_sensor_count * 100, 2)
        summary['sensor_count_passing_sda'] = int(total_sensor_count_pass_sda)
        summary['total_sensor_count'] = total_sensor_count

    return summary, summary_grid


def well_annual_daylight(
        results: Union[str, AnnualDaylight], daylight_hours: list, sub_folder,
        grids_filter: str = '*', states_schedule: dict = None):
    """Calculate credits for WELL L06.

    Args:
        results: Path to results folder or a Results class object.
        daylight_hours: Schedule of daylight hours used for EN 17037
        sub_folder: Relative path for a subfolder to write the output.
        grids_filter: The name of a grid or a pattern to filter the grids.
            Defaults to '*'.
        states_schedule: A custom dictionary of shading states. In case this is
            left empty, the function will calculate a shading schedule by using
            the shade_transmittance input. If a states schedule is provided it
            will check that it is complying with the 2% rule. Defaults to None.

    Returns:
        Tuple:
        -   well_summary: Summary of WELL analysis.
        -   ies_lm_summary: Summary of IES LM analysis.
        -   ies_lm_summary_grid: Summary of IES LM analysis for each grid.
        -   da_grids: List of daylight autonomy values for each grid. Each item
                in the list is a NumPy array of DA values.
        -   states_schedule: A dictionary of annual shading schedules for each
                aperture group.
        -   fail_to_comply: A dictionary with the hoys where the 2% rule failed.
        -   grids_info: Grid information.
    """
    schedule = occupancy_schedule_8_to_6(as_list=True)

    if not isinstance(results, AnnualDaylight):
        results = AnnualDaylight(results, schedule=schedule, cache_arrays=False)
    else:
        # set schedule to default leed schedule
        results.schedule = schedule

    occ_mask = results.occ_mask
    total_occ = results.total_occ

    grids_info = results._filter_grids(grids_filter=grids_filter)

    if not states_schedule:
        states_schedule, fail_to_comply, shd_trans_dict = dynamic_schedule_direct_illuminance(
            results, grids_filter=grids_filter, use_states=True)
    else:
        raise NotImplementedError(
            'Custom input for argument states_schedule is not yet implemented.'
        )

    # check to see if there is a HBJSON with sensor grid meshes for areas
    grid_areas, units_conversion = [], 1
    for base_file in Path(results.folder).parent.iterdir():
        if base_file.suffix in ('.hbjson', '.hbpkl'):
            hb_model = Model.from_file(base_file)
            units_conversion = conversion_factor_to_meters(hb_model.units)
            filt_grids = _filter_by_pattern(
                hb_model.properties.radiance.sensor_grids, filter=grids_filter)
            for s_grid in filt_grids:
                if s_grid.mesh is not None:
                    grid_areas.append(s_grid.mesh.face_areas)
            grid_areas = [np.array(grid) for grid in grid_areas]
            break
    if not grid_areas:
        grid_areas = [None] * len(grids_info)

    # spatial daylight autonomy
    l06_da_grids = []
    l06_pass_sda_grids = []
    l06_pass_sda_blinds_up_grids = []
    l06_pass_sda_blinds_down_grids = []
    l01_da_grids = []
    l01_pass_sda_grids = []
    l01_pass_sda_blinds_up_grids = []
    l01_pass_sda_blinds_down_grids = []
    for grid_info in grids_info:
        light_paths = []
        for lp in grid_info['light_path']:
            for _lp in lp:
                if _lp == '__static_apertures__' and len(lp) > 1:
                    pass
                else:
                    light_paths.append(_lp)
        base_zero_array = np.apply_along_axis(filter_array, 1, np.zeros(
            (grid_info['count'], len(results.sun_up_hours))), occ_mask)
        arrays_blinds_up = [base_zero_array.copy()]
        arrays_blinds_down = [base_zero_array.copy()]
        # combine total array for all light paths
        array = results._array_from_states(grid_info, states=states_schedule, zero_array=True)
        array = np.apply_along_axis(filter_array, 1, array, occ_mask)

        for light_path in light_paths:
            # do an extra pass to calculate with blinds always up or down
            if light_path != '__static_apertures__':
                array_blinds_up = results._get_array(
                    grid_info, light_path, state=0, res_type='total')
                array_filter = np.apply_along_axis(
                    filter_array, 1, array_blinds_up, occ_mask)
                arrays_blinds_up.append(array_filter)
                array_blinds_down = results._get_array(
                    grid_info, light_path, state=1, res_type='total')
                array_filter = np.apply_along_axis(
                    filter_array, 1, array_blinds_down, occ_mask)
                arrays_blinds_down.append(array_filter)
            else:
                static_array = results._get_array(
                    grid_info, light_path, state=0, res_type='total')
                array_filter = np.apply_along_axis(
                    filter_array, 1, static_array, occ_mask)
                arrays_blinds_up.append(array_filter)
                arrays_blinds_down.append(array_filter)

        array_blinds_up = sum(arrays_blinds_up)
        array_blinds_down = sum(arrays_blinds_down)
        # calculate da per grid
        da_grid = da_array2d(array, total_occ=total_occ, threshold=300)

        l06_da_grids.append(da_grid)
        da_blinds_up_grid = da_array2d(
            array_blinds_up, total_occ=total_occ, threshold=300)
        da_blinds_down_grid = da_array2d(
            array_blinds_down, total_occ=total_occ, threshold=300)
        # calculate sda per grid
        l06_pass_sda_grids.append(da_grid >= 50)
        l06_pass_sda_blinds_up_grids.append(da_blinds_up_grid >= 50)
        l06_pass_sda_blinds_down_grids.append(da_blinds_down_grid >= 50)

        array_blinds_up = sum(arrays_blinds_up)
        array_blinds_down = sum(arrays_blinds_down)
        # calculate da per grid
        da_grid = da_array2d(array, total_occ=total_occ, threshold=200)

        l01_da_grids.append(da_grid)
        da_blinds_up_grid = da_array2d(
            array_blinds_up, total_occ=total_occ, threshold=200)
        da_blinds_down_grid = da_array2d(
            array_blinds_down, total_occ=total_occ, threshold=200)
        # calculate sda per grid
        l01_pass_sda_grids.append(da_grid >= 40)
        l01_pass_sda_blinds_up_grids.append(da_blinds_up_grid >= 40)
        l01_pass_sda_blinds_down_grids.append(da_blinds_down_grid >= 40)

    # create summaries for all grids and each grid individually
    l06_ies_lm_summary, l06_ies_lm_summary_grid = _well_summary(
        l06_pass_sda_grids, grids_info, grid_areas,
        l06_pass_sda_blinds_up_grids, l06_pass_sda_blinds_down_grids)
    l01_ies_lm_summary, l01_ies_lm_summary_grid = _well_summary(
        l01_pass_sda_grids, grids_info, grid_areas,
        l01_pass_sda_blinds_up_grids, l01_pass_sda_blinds_down_grids)

    sub_folder = Path(sub_folder)
    en17037_folder = en17037_to_folder(
        results, schedule=daylight_hours,
        sub_folder=sub_folder.joinpath('en17037'))

    l06_well_summary = []
    l01_well_summary = []
    l06_well_summary_ies_lm = {}
    l06_well_summary_en17037 = {}
    l01_well_summary_ies_lm = {}
    l01_well_summary_en17037 = {}
    l06_well_summary_ies_lm['method'] = 'IES LM-83-12'
    l06_well_summary_en17037['method'] = 'EN 17037'
    l01_well_summary_ies_lm['method'] = 'IES LM-83-12'
    l01_well_summary_en17037['method'] = 'EN 17037'

    l06_combined_da_target = []
    l06_combined_da_minimum = []
    l01_combined_da_target = []
    for grid_info in grids_info:
        l06_grid_da_target = np.loadtxt(en17037_folder.joinpath('da', 'target_illuminance_300', f'{grid_info["full_id"]}.da'))
        l06_grid_da_minimum = np.loadtxt(en17037_folder.joinpath('da', 'minimum_illuminance_100', f'{grid_info["full_id"]}.da'))
        l06_combined_da_target.append(l06_grid_da_target >= 50)
        l06_combined_da_minimum.append(l06_grid_da_minimum >= 50)

        array = results._array_from_states(
            grid_info, res_type='total', zero_array=True)
        if np.any(array):
            array = np.apply_along_axis(
                filter_array, 1, array, occ_mask)
        l01_grid_da_target = da_array2d(array, total_occ=4380, threshold=200)
        l01_combined_da_target.append(l01_grid_da_target >= 50)

    l06_combined_sda_target_illuminance = np.concatenate(l06_combined_da_target).mean() * 100
    l06_combined_sda_minimum_illuminance = np.concatenate(l06_combined_da_minimum).mean() * 100

    l01_combined_sda_target_illuminance = np.concatenate(l01_combined_da_target).mean() * 100

    if l01_combined_sda_target_illuminance > 30:
        l01_well_summary_en17037['comply'] = True
    else:
        l01_well_summary_en17037['comply'] = False

    if l06_combined_sda_target_illuminance > 50 and l06_combined_sda_minimum_illuminance > 95:
        l06_well_summary_en17037['points'] = 2
    elif l06_combined_sda_target_illuminance > 50:
        l06_well_summary_en17037['points'] = 1
    else:
        l06_well_summary_en17037['points'] = 0

    # credits
    if not fail_to_comply:
        if l06_ies_lm_summary['sda'] >= 75:
            l06_ies_lm_summary['credits'] = 3
            l06_well_summary_ies_lm['points'] = 2
        elif l06_ies_lm_summary['sda'] >= 55:
            l06_ies_lm_summary['credits'] = 2
            l06_well_summary_ies_lm['points'] = 1
        elif l06_ies_lm_summary['sda'] >= 40:
            l06_ies_lm_summary['credits'] = 1
            l06_well_summary_ies_lm['points'] = 0
        else:
            l06_ies_lm_summary['credits'] = 0
            l06_well_summary_ies_lm['points'] = 0

        if all(grid_summary['sda'] >= 55 for grid_summary in l06_ies_lm_summary_grid.values()):
            if l06_ies_lm_summary['credits'] <= 2:
                l06_ies_lm_summary['credits'] += 1
            else:
                l06_ies_lm_summary['credits'] = 'Exemplary performance'

        if l01_ies_lm_summary['sda'] >= 30:
            l01_well_summary_ies_lm['comply'] = True
        else:
            l01_well_summary_ies_lm['comply'] = False

        l06_well_summary_ies_lm['sda'] = l06_ies_lm_summary['sda']
        l01_well_summary_ies_lm['sda'] = l01_ies_lm_summary['sda']
    else:
        l06_ies_lm_summary['credits'] = 0
        fail_to_comply_rooms = ', '.join(list(fail_to_comply.keys()))
        note = (
            '0 credits have been awarded. The following sensor grids have at '
            'least one hour where 2% of the floor area receives direct '
            f'illuminance of 1000 lux or more: {fail_to_comply_rooms}.'
        )
        l06_ies_lm_summary['note'] = note
        l06_well_summary_ies_lm['points'] = 0
        l06_well_summary_ies_lm['sda'] = l06_ies_lm_summary['sda']
        l06_well_summary_ies_lm['note'] = note

        l01_ies_lm_summary['note'] = note
        l01_well_summary_ies_lm['comply'] = False
        l01_well_summary_ies_lm['sda'] = l01_ies_lm_summary['sda']
        l01_well_summary_ies_lm['note'] = note

    l06_well_summary_ies_lm['total_floor_area'] = sum(np.sum(arr) for arr in grid_areas)
    l01_well_summary_ies_lm['total_floor_area'] = sum(np.sum(arr) for arr in grid_areas)

    # convert to datacollection
    def to_datacollection(aperture_group: str, values: np.ndarray):
        # convert values to 0 and 1 (0 = no shading, 1 = shading)
        header = Header(data_type=GenericType(aperture_group, ''), unit='',
                        analysis_period=AnalysisPeriod())
        hourly_data = HourlyContinuousCollection(header=header, values=values)
        return hourly_data.to_dict()

    states_schedule = {k:to_datacollection(k, v['schedule']) for k, v in states_schedule.to_dict().items()}

    ies_lm_folder = sub_folder.joinpath('ies_lm')
    ies_lm_folder.mkdir(parents=True, exist_ok=True)

    l06_ies_lm_folder = ies_lm_folder.joinpath('l06_ies_lm_summary')
    l01_ies_lm_folder = ies_lm_folder.joinpath('l01_ies_lm_summary')

    l06_ies_lm_folder.mkdir(parents=True, exist_ok=True)
    l01_ies_lm_folder.mkdir(parents=True, exist_ok=True)

    ies_lm_summary_file = l06_ies_lm_folder.joinpath('ies_lm_summary.json')
    ies_lm_summary_file.write_text(json.dumps(l06_ies_lm_summary, indent=2))
    ies_lm_summary_file = l01_ies_lm_folder.joinpath('ies_lm_summary.json')
    ies_lm_summary_file.write_text(json.dumps(l01_ies_lm_summary, indent=2))

    ies_lm_summary_grid_file = l06_ies_lm_folder.joinpath('ies_lm_summary_grid.json')
    ies_lm_summary_grid_file.write_text(json.dumps(l06_ies_lm_summary_grid, indent=2))
    ies_lm_summary_grid_file = l01_ies_lm_folder.joinpath('ies_lm_summary_grid.json')
    ies_lm_summary_grid_file.write_text(json.dumps(l01_ies_lm_summary_grid, indent=2))

    states_schedule_file = l06_ies_lm_folder.joinpath('states_schedule.json')
    states_schedule_file.write_text(json.dumps(states_schedule))
    states_schedule_file = l01_ies_lm_folder.joinpath('states_schedule.json')
    states_schedule_file.write_text(json.dumps(states_schedule))

    grids_info_file = l06_ies_lm_folder.joinpath('grids_info.json')
    grids_info_file.write_text(json.dumps(grids_info, indent=2))
    grids_info_file = l01_ies_lm_folder.joinpath('grids_info.json')
    grids_info_file.write_text(json.dumps(grids_info, indent=2))

    for (da, grid_info) in \
        zip(l06_da_grids, grids_info):
        grid_id = grid_info['full_id']
        da_file = l06_ies_lm_folder.joinpath('results', 'da', f'{grid_id}.da')
        da_file.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(da_file, da, fmt='%.2f')
    for (da, grid_info) in \
        zip(l01_da_grids, grids_info):
        grid_id = grid_info['full_id']
        da_file = l01_ies_lm_folder.joinpath('results', 'da', f'{grid_id}.da')
        da_file.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(da_file, da, fmt='%.2f')

    sda_pf_folder = ies_lm_folder.joinpath('pass_fail')
    sda_pf_folder.mkdir(parents=True, exist_ok=True)
    for l01_pf, grid_info in zip(l01_pass_sda_grids, grids_info):
        grid_id = grid_info['full_id']
        l01_sda_pf_folder = sda_pf_folder.joinpath('L01')
        l01_sda_pf_folder.mkdir(parents=True, exist_ok=True)
        l01_pf_file = l01_sda_pf_folder.joinpath(f'{grid_id}.pf')
        l01_pf = l01_pf.astype(int)
        np.savetxt(l01_pf_file, l01_pf, fmt='%d')
        grids_info_file = l01_sda_pf_folder.joinpath('grids_info.json')
        grids_info_file.write_text(json.dumps(grids_info, indent=2))
    for l06_pf, grid_info in zip(l06_pass_sda_grids, grids_info):
        grid_id = grid_info['full_id']
        l06_sda_pf_folder = sda_pf_folder.joinpath('L06')
        l06_sda_pf_folder.mkdir(parents=True, exist_ok=True)
        l06_pf_file = l06_sda_pf_folder.joinpath(f'{grid_id}.pf')
        l06_pf = l06_pf.astype(int)
        np.savetxt(l06_pf_file, l06_pf, fmt='%d')
        grids_info_file = l06_sda_pf_folder.joinpath('grids_info.json')
        grids_info_file.write_text(json.dumps(grids_info, indent=2))

    da_grids_info_file = l06_ies_lm_folder.joinpath(
        'results', 'da', 'grids_info.json')
    da_grids_info_file.write_text(json.dumps(grids_info, indent=2))
    da_grids_info_file = l01_ies_lm_folder.joinpath(
        'results', 'da', 'grids_info.json')
    da_grids_info_file.write_text(json.dumps(grids_info, indent=2))

    states_schedule_err_file = \
        l06_ies_lm_folder.joinpath('states_schedule_err.json')
    states_schedule_err_file.write_text(json.dumps(fail_to_comply))
    states_schedule_err_file = \
        l01_ies_lm_folder.joinpath('states_schedule_err.json')
    states_schedule_err_file.write_text(json.dumps(fail_to_comply))

    l06_well_summary.append(l06_well_summary_ies_lm)
    l06_well_summary.append(l06_well_summary_en17037)
    well_summary_file = sub_folder.joinpath('l06_well_summary.json')
    well_summary_file.write_text(json.dumps(l06_well_summary, indent=2))
    l01_well_summary.append(l01_well_summary_ies_lm)
    l01_well_summary.append(l01_well_summary_en17037)
    well_summary_file = sub_folder.joinpath('l01_well_summary.json')
    well_summary_file.write_text(json.dumps(l01_well_summary, indent=2))

    return (l06_well_summary, l01_well_summary, states_schedule, fail_to_comply, grids_info)
