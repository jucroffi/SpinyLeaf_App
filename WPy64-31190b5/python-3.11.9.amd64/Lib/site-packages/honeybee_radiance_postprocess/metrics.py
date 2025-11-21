"""Functions to calculate various metrics for 1D and 2D NumPy arrays."""
from typing import Tuple, Union
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from .util import check_array_dim

is_cpu = not is_gpu


def da_array2d(
        array: np.ndarray, total_occ: int = None, threshold: float = 300) -> np.ndarray:
    """Calculate daylight autonomy for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not
            given any input the number of occupied hours will be found by the
            array shape, i.e., it is assumed that the array is filtered by
            occupied hours.
        threshold: Threshold value for daylight autonomy. Default: 300.

    Returns:
        A 1-dimensional NumPy array with the daylight autonomy for each row in
        the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    da = np.sum(array >= threshold, axis=1) / total_occ * 100

    return da


def da_array1d(
        array: np.ndarray, total_occ: int = None,
        threshold: float = 300) -> np.float64:
    """Calculate daylight autonomy for a 1D NumPy array.

    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for daylight autonomy. Default: 300.

    Returns:
        A NumPy float of the daylight autonomy.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64((array >= threshold).sum() / total_occ * 100)


def cda_array2d(
        array: np.ndarray, total_occ: int = None,
        threshold: float = 300) -> np.ndarray:
    """Calculate continuos daylight autonomy for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for continuos daylight autonomy. Default: 300.

    Returns:
        A 1-dimensional NumPy array with the continuos daylight autonomy for
        each row in the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    if is_cpu:
        cda = np.apply_along_axis(
                cda_array1d, 1, array, total_occ=total_occ, threshold=threshold)
    else:
        cda = np.where(array >= threshold, 1, array / threshold).sum(axis=1) / total_occ * 100

    return cda


def cda_array1d(
        array: np.ndarray, total_occ: int = None,
        threshold: float = 300) -> np.float64:
    """Calculate continuos daylight autonomy for a 1D NumPy array.

    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        threshold: Threshold value for continuos daylight autonomy. Default: 300.

    Returns:
        A NumPy float of the continuos daylight autonomy.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64(
        np.where(array >= threshold, 1, array / threshold).sum() / total_occ * 100)


def udi_array2d(
        array: np.ndarray, total_occ: int = None, min_t: float = 100,
        max_t: float = 3000) -> np.ndarray:
    """Calculate useful daylight illuminance for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        min_t: Minimum threshold for useful daylight illuminance. Default: 100.
        max_t: Maximum threshold for useful daylight illuminance. Default: 3000.

    Returns:
        A 1-dimensional NumPy array with the useful daylight illuminance for
        each row in the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    if is_cpu:
        udi = np.apply_along_axis(
                udi_array1d, 1, array, total_occ=total_occ, min_t=min_t, max_t=max_t)
    else:
        udi = ((array >= min_t) & (array <= max_t)).sum(axis=1) / total_occ * 100

    return udi


def udi_array1d(
        array: np.ndarray, total_occ: int = None, min_t: float = 100,
        max_t: float = 3000) -> np.float64:
    """Calculate useful daylight illuminance for a 1D NumPy array.

    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        min_t: Minimum threshold for useful daylight illuminance. Default: 100.
        max_t: Maximum threshold for useful daylight illuminance. Default: 3000.

    Returns:
        A NumPy float of the useful daylight illuminance.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64(((array >= min_t) & (array <= max_t)).sum() / total_occ * 100)


def udi_lower_array2d(
        array: np.ndarray, total_occ: int = None, min_t: float = 100,
        sun_down_occ_hours: int = 0) -> np.ndarray:
    """Calculate lower than useful daylight illuminance for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        min_t: Minimum threshold for useful daylight illuminance. Default: 100.
        sun_down_occ_hours: Number of occupied hours where the sun is down.

    Returns:
        A 1-dimensional NumPy array with the lower than useful daylight
        illuminance for each row in the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    if is_cpu:
        udi = np.apply_along_axis(
                udi_lower_array1d, 1, array, total_occ=total_occ, min_t=min_t,
                sun_down_occ_hours=sun_down_occ_hours)
    else:
        if min_t == 0:
            return np.zeros(array.shape[0], dtype=np.float32)

        udi = ((array < min_t).sum(axis=1) + sun_down_occ_hours) / total_occ * 100

    return udi


def udi_lower_array1d(
        array: np.ndarray, total_occ: int = None, min_t: float = 100,
        sun_down_occ_hours: int = 0) -> np.float64:
    """Calculate lower than useful daylight illuminance for a 1D NumPy array.

    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        min_t: Minimum threshold for useful daylight illuminance. Default: 100.
        sun_down_occ_hours: Number of occupied hours where the sun is down.

    Returns:
        A NumPy float of the lower than useful daylight illuminance.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    if min_t == 0:
        return np.float64(0)

    return np.float64(((array < min_t).sum() + sun_down_occ_hours) / total_occ * 100)


def udi_upper_array2d(
        array: np.ndarray, total_occ: int = None,
        max_t: float = 3000) -> np.ndarray:
    """Calculate higher than useful daylight illuminance for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        max_t: Maximum threshold for useful daylight illuminance. Default: 3000.

    Returns:
        A 1-dimensional NumPy array with the higher than useful daylight
        illuminance for each row in the input array.
    """
    check_array_dim(array, 2)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.shape[1]

    if is_cpu:
        udi = np.apply_along_axis(
                udi_upper_array1d, 1, array, total_occ=total_occ, max_t=max_t)
    else:
        udi = (array > max_t).sum(axis=1) / total_occ * 100

    return udi


def udi_upper_array1d(
        array: np.ndarray, total_occ: int = None,
        max_t: float = 3000) -> np.float64:
    """Calculate higher than useful daylight illuminance for a 1D NumPy array.

    Args:
        array: A 1D NumPy array.
        total_occ: Integer indicating the number of occupied hours. If not given any
            input the number of occupied hours will be found by the array shape.
        max_t: Maximum threshold for higher than useful daylight illuminance.
            Default: 3000.

    Returns:
        A NumPy float of the higher than useful daylight illuminance.
    """
    check_array_dim(array, 1)
    if total_occ is None:
        # set total_occ to number of columns in array
        total_occ = array.size

    return np.float64((array > max_t).sum() / total_occ * 100)


def sda_array2d(
        array: np.ndarray, target_time: float = 50, threshold: float = 300,
        total_occ: int = None) -> np.ndarray:
    """Calculate spatial daylight autonomy for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        target_time: A minimum threshold of occupied time (eg. 50% of the
            time), above which a given sensor passes and contributes to the
            spatial daylight autonomy. Defaults to 50.
        threshold: Threshold value for daylight autonomy. Default: 300.
        total_occ: Integer indicating the number of occupied hours. If not
            given any input the number of occupied hours will be found by the
            array shape, i.e., it is assumed that the array is filtered by
            occupied hours.

    Returns:
        A NumPy float of the sDA as a percentage (decimal)
    """
    da = np.sum(array >= threshold, axis=1) / total_occ * 100
    sda = (da >= target_time).mean()

    return sda


def ase_array2d(
        array: np.ndarray, occ_hours: int = 250,
        direct_threshold: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate annual sunlight exposure for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        occ_hours: The number of occupied hours that cannot receive more than
            the direct_threshold. Defaults to 250.
        direct_threshold: The threshold that determines if a sensor is overlit.
            Defaults to 1000.

    Returns:
        A Tuple with two values.

        -   ase: NumPy float of the ASE as a percentage (decimal).

        -   h_above: 1D NumPy array of the number of hours above the direct
            threshold.
    """
    check_array_dim(array, 2)
    h_above = (array > direct_threshold).sum(axis=1)
    ase = (h_above >= occ_hours).sum() / array.shape[0] * 100

    return ase, h_above


def average_values_array2d(
        array: np.ndarray, full_length: int = 8760) -> np.ndarray:
    """Calculate average values for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        full_length: Integer to use as divisor.

    Returns:
        A 1-dimensional NumPy array with the average value for each row in the
        input array.
    """
    check_array_dim(array, 2)

    avg_values = array.sum(axis=1) / full_length

    return avg_values


def average_values_array1d(
        array: np.ndarray, full_length: int = 8760) -> np.float64:
    """Calculate average value for a 1D NumPy array.

    Args:
        array: A 1D NumPy array.
        full_length: Integer to use as divisor.

    Returns:
        A NumPy float of the average value.
    """
    check_array_dim(array, 1)

    return array.sum() / full_length


def cumulative_values_array2d(
        array: np.ndarray, timestep: int = 1, t_step_multiplier: float = 1
        ) -> np.ndarray:
    """Calculate cumulative values for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        timestep: Integer for the timestep of the analysis.
        t_step_multiplier: A value that will be multiplied with the timestep.

    Returns:
        A 1-dimensional NumPy array with the cumulative value for each row in
        the input array.
    """
    check_array_dim(array, 2)

    cumulative_values = array.sum(axis=1) / (timestep * t_step_multiplier)

    return cumulative_values


def cumulative_values_array1d(
        array: np.ndarray, timestep: int = 1, t_step_multiplier: float = 1
        ) -> np.float64:
    """Calculate daylight autonomy for a 1D NumPy array.

    Args:
        array: A 1D NumPy array.
        timestep: Integer for the timestep of the analysis.
        t_step_multiplier: A value that will be multiplied with the timestep.

    Returns:
        A NumPy float of the cumulative value.
    """
    check_array_dim(array, 1)

    return array.sum() / (timestep * t_step_multiplier)


def peak_values_array2d(
        array: np.ndarray, coincident: bool = False
        ) -> Tuple[np.ndarray, Union[int, None]]:
    """Calculate peak values for a 2D NumPy array.

    Args:
        array: A 2D NumPy array.
        coincident: Boolean to indicate whether output values represent the
            peak value for each sensor throughout the entire analysis (False)
            or they represent the highest overall value across each sensor grid
            at a particular timestep (True).

    Returns:
        A 1-dimensional NumPy array with the peak value for each row in the
        input array, and the index of the maximum value representing the
        timestep in the array with the largest value.
    """
    check_array_dim(array, 2)

    max_i = None
    if coincident:
        array_summed = array.sum(axis=0)
        if np.any(array_summed):
            max_i = np.argmax(array_summed)
            peak_values = array[:, max_i]
        else:
            peak_values = np.zeros(array.shape[0])
    else:
        if np.any(array):
            peak_values = np.amax(array, axis=1)
        else:
            peak_values = np.zeros(array.shape[0])

    return peak_values, max_i
