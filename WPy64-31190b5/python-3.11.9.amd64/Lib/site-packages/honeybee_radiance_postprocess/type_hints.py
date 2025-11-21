"""Type hints for honeybee-radiance-postprocess."""
from typing import Tuple, List
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from ladybug.datacollection import HourlyContinuousCollection


annual_metric = Tuple[List[np.ndarray], List[dict]]
annual_daylight_metrics = Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[dict]
]
annual_irradiance_metrics = Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[dict]
]
spatial_daylight_autonomy = Tuple[List[np.float64], List[dict]]
annual_sunlight_exposure = Tuple[List[np.float64], List[np.ndarray], List[dict]]
total = Tuple[List[np.ndarray], List[dict]]
point_in_time = Tuple[List[np.ndarray], List[dict]]
average_values = Tuple[List[np.ndarray], List[dict]]
median_values = Tuple[List[np.ndarray], List[dict]]
cumulative_values = Tuple[List[np.ndarray], List[dict]]
peak_values = Tuple[List[np.ndarray], List[dict]]
annual_data = Tuple[List[List[HourlyContinuousCollection]], List[dict], dict]
annual_uniformity_ratio = \
    Tuple[List[float], List[HourlyContinuousCollection], List[dict]]
