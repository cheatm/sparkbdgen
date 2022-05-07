from typing import Callable
import numpy as np


DISTRIBUTION_FUNC = Callable[[int], np.ndarray]
ARRAY_MAPPING = Callable[[np.ndarray], np.ndarray]