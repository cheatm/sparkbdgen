from typing import Callable
import numpy as np


__all__ = ["DISTRIBUTION_FUNC", "ARRAY_MAPPING", "CLUSTER_SHIFT"]

DISTRIBUTION_FUNC = Callable[[int], np.ndarray]
ARRAY_MAPPING = Callable[[np.ndarray], np.ndarray]
CLUSTER_SHIFT = Callable[[int, int], np.ndarray]