import numpy as np
from numpy import random
from functools import partial
from .types import DISTRIBUTION_FUNC


def constant(value: np.float64) ->DISTRIBUTION_FUNC:
    return lambda size: np.zeros(size) + value


def uniform(minimum: np.float64=0, maximum: np.float64=1) -> DISTRIBUTION_FUNC:
    return partial(random.uniform, minimum, maximum)


def normal(loc=0.0, scale=1.0) -> DISTRIBUTION_FUNC:
    return partial(random.normal, loc, scale)




