"""Shared functions for post-processing annual results."""
from typing import Union
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ladybug.analysisperiod import AnalysisPeriod
from .util import filter_array

is_cpu = not is_gpu


def occupancy_schedule_8_to_6(
        timestep: int = 1, as_list: bool = False) -> Union[np.ndarray, list]:
    """Create an occupancy schedule for LEED (8 am to 6 pm).

    Args:
        timestep: An integer value noting the number of timesteps per hour.
            Defaults to 1.
        as_list: Boolean toggle to output the schedule as a Python list instead
            of a NumPy array. Defaults to False.

    Returns:
        A schedule as an array or list.
    """
    full_analysis_period = np.array(AnalysisPeriod(timestep=timestep).hoys)
    analysis_period = np.array(AnalysisPeriod(st_hour=8, end_hour=17, timestep=timestep).hoys)
    schedule = np.zeros(8760 * timestep).astype(int)
    hours = np.where(np.isin(full_analysis_period, analysis_period))[0]
    schedule[hours] = 1
    if as_list:
        schedule = schedule.tolist()

    return schedule


def schedule_to_hoys(
        schedule: Union[list, np.ndarray],
        sun_up_hours: Union[list, np.ndarray] = None, as_list: bool = False
        ) -> Union[np.ndarray, list]:
    """Convert a schedule to hoys.

    Args:
        schedule: A list of 8760 values for the occupancy schedule.
        sun_up_hours: An optional list of sun up hours as integers. If sun up
            hours are provided the function will exclude all values from the
            schedule that are not among the sun up hours. Defaults to None.
        as_list: Boolean toggle to output the schedule as a Python list instead
            of a NumPy array. Defaults to False.

    Returns:
        An array or list of occupancy expressed as hoys.
    """
    assert len(schedule) == 8760
    if not isinstance(schedule, np.ndarray):
        schedule = np.array(schedule).astype(int)

    hours = np.arange(0, 8760, 1)
    if sun_up_hours:
        sun_up_hours = np.array(sun_up_hours).astype(int)
        mask = np.ones(schedule.size, dtype=bool)
        mask[sun_up_hours] = False
        schedule[mask] = 0

    occ_hoys = filter_array(hours, np.array(schedule))

    if as_list:
        occ_hoys = occ_hoys.tolist()

    return occ_hoys
