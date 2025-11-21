"""Post-processing utility functions."""
from typing import Tuple
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from honeybee_radiance.writer import _filter_by_pattern

is_cpu = not is_gpu


def binary_mtx_dimension(filepath: str) -> Tuple[int, int, int, int, str]:
    """Return binary Radiance matrix dimensions if exist.

    This function returns NROWS, NCOLS, NCOMP and number of header lines including the
    white line after last header line.

    Args:
        filepath: Full path to Radiance file.

    Returns:
        nrows, ncols, ncomp, line_count, fmt
    """
    try:
        inf = open(filepath, 'rb', encoding='utf-8')
    except Exception:
        inf = open(filepath, 'rb')
    try:
        first_line = next(inf).rstrip().decode('utf-8')
        if first_line[:10] != '#?RADIANCE':
            error_message = (
                f'File with Radiance header must start with #?RADIANCE not '
                f'{first_line}.'
            )
            raise ValueError(error_message)

        header_lines = [first_line]
        nrows = ncols = ncomp = None
        for line in inf:
            line = line.rstrip().decode('utf-8')
            header_lines.append(line)
            if line[:6] == 'NROWS=':
                nrows = int(line.split('=')[-1])
            if line[:6] == 'NCOLS=':
                ncols = int(line.split('=')[-1])
            if line[:6] == 'NCOMP=':
                ncomp = int(line.split('=')[-1])
            if line[:7] == 'FORMAT=':
                fmt = line.split('=')[-1]
                break

        if not nrows or not ncols:
            error_message = (
                f'NROWS or NCOLS was not found in the Radiance header. NROWS '
                f'is {nrows} and NCOLS is {ncols}. The header must have both '
                f'elements.'
            )
            raise ValueError(error_message)
        return nrows, ncols, ncomp, len(header_lines) + 1, fmt
    finally:
        inf.close()


def check_array_dim(array: np.ndarray, dim: int):
    """Check NumPy array dimension.

    Args:
        array: A NumPy array.
        dim: The dimension to check against.
    """
    assert array.ndim == dim, \
        f'Expected {dim}-dimensional array. Dimension of array is {array.ndim}'


def filter_array2d(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Filter a NumPy array by a masking array. The array will be passed as is
    if the mask is None.

    Args:
        array: A NumPy array to filter.
        mask: A NumPy array of ones/zeros or True/False.

    Returns:
        A filtered NumPy array.
    """
    if is_cpu:
        array_filter = np.apply_along_axis(
            filter_array, 1, array, mask=mask)
    else:
        mask = np.asarray(mask).astype(bool)
        array_filter = array[:, mask]

    return array_filter


def filter_array(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Filter a NumPy array by a masking array. The array will be passed as is
    if the mask is None.

    Args:
        array: A NumPy array to filter.
        mask: A NumPy array of ones/zeros or True/False.

    Returns:
        A filtered NumPy array.
    """
    if mask is not None:
        return array[mask.astype(bool)]
    return array


def hoys_mask(sun_up_hours: list, hoys: list) -> np.ndarray:
    """Create a NumPy masking array from a list of hoys.

    Args:
        sun_up_hours: A list of sun up hours.
        hoys: A list hoys to select.

    Returns:
        A NumPy array of booleans.
    """
    if len(hoys) != 0:
        hoys_mask = np.where(np.isin(np.array(sun_up_hours), np.array(hoys)), True, False)
        return hoys_mask


def array_memory_size(
    sensors: int, sun_up_hours: int, ncomp: int = None,
    dtype: np.dtype = np.float32, gigabyte: bool = True) -> float:
    """Calculate the memory size of an array before creating or loading an
    array.

    Args:
        sensors: Number of sensors in the array.
        sun_up_hours: Number of sun up hours in the array.
        ncomp: Optional number of components for each element in the array,
            e.g., if the data is in RGB format then this value must be set
            to 3. Defaults to None.
        dtype: The data type of the array. Defaults to np.float32.
        gigabyte: Boolean toggle to output the memory size in gigabytes.
            Defaults to True.

    Returns:
        float: The memory size of an array.
    """
    # check if dtype is valid
    dtypes = tuple(np.sctypes['float'])
    if not isinstance(dtype, dtypes):
        try:
            dtype = dtype()
        except TypeError as err:
            error_message = (
                f'Unable to instantiate input dtype. Expected any of the '
                f'following: {dtypes}. Received: {type(dtype)}.'
            )
            raise TypeError(error_message) from err

    # calculate memory size
    size = sensors * sun_up_hours * dtype.itemsize
    if ncomp:
        size *= ncomp
    if gigabyte:
        size /= (1024 ** 3)

    return size


def recursive_dict_merge(dict_1: dict, dict_2: dict):
    """Recursive merging of two dictionaries.

    Args:
        dict_1: Original dictionary.
        dict_2: Dictionary to merge with dict_1.
    """
    for k in dict_2:
        if (k in dict_1 and isinstance(dict_1[k], dict) and
            isinstance(dict_2[k], dict)):
            recursive_dict_merge(dict_1[k], dict_2[k])
        else:
            dict_1[k] = dict_2[k]


def _filter_grids_by_pattern(grids_info, filter_pattern):
    """Filter grids_info by a pattern.

    Args:
        grids_info: Grid information.
        filter_pattern: Pattern to filter grids by.

    Returns:
        A list of filtered grids.
    """
    grids = _filter_by_pattern(grids_info, filter=filter_pattern)

    return grids


def get_delimiter(delimiter_input):
    if delimiter_input == 'tab' or delimiter_input == '\t':
        return '\t'
    elif delimiter_input == 'space' or delimiter_input == ' ':
        return ' '
    elif delimiter_input == 'comma' or delimiter_input == ',':
        return ','
    elif delimiter_input == 'semicolon' or delimiter_input == ';':
        return ';'
    else:
        raise ValueError(f'Unsupported delimiter: {delimiter_input}')
