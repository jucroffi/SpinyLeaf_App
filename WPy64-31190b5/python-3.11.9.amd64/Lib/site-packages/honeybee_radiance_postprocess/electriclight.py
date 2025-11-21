"""Functions for post-processing daylight outputs into electric lighting schedules."""
from typing import List
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np


def array_to_dimming_fraction(
        array: np.ndarray, su_pattern: List[int], setpt: float, m_pow: float,
        m_lgt: float, off_m: float) -> list:
    """Compute hourly dimming fractions for a given result file."""
    fract_dim = (setpt - array) / (setpt - m_lgt)
    par_dim = fract_dim + ((1 - fract_dim) * m_pow)
    su_values = np.where(array > setpt, 0 if off_m else m_pow,
                            (np.where(array <= m_lgt, 1, par_dim))).sum(axis=0)
    su_values = su_values / array.shape[0]

    dim_fract = np.ones(8760)
    dim_fract[su_pattern] = su_values

    return dim_fract
