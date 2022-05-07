from typing import List, Callable
from unittest import result
import numpy as np
from .types import ARRAY_MAPPING
from .types import DISTRIBUTION_FUNC


def rotation_matrix_3d(theta: np.float64, axis: int) -> result:
    sint = np.sin(theta)
    cost = np.cos(theta)
    result = np.identity(3)
    result[(axis+1)%3, (axis+1)%3] = cost
    result[(axis+2)%3, (axis+1)%3] = sint
    result[(axis+1)%3, (axis+2)%3] = -sint
    result[(axis+2)%3, (axis+2)%3] = cost
    return result


class LinearMapping(object):

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
    
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, array.T).T

    @classmethod
    def identity(cls, size: int):
        if size < 1:
            raise ValueError(f"Invalid dimension: {size}")
        return cls(np.identity(size, np.float64))

    @classmethod
    def rotation2D(cls, theta: np.float64):
        sint = np.sin(theta)
        cost = np.cos(theta)
        return cls(np.array([
            [cost,-sint],
            [sint,cost]
        ]))

    @classmethod
    def rotation3D(cls, roll: np.float64=0, pitch: np.float64=0, yaw: np.float64=0):
        matrix = np.identity(3, np.float64)
        for axis, theta in enumerate([roll, pitch, yaw]):
            if theta != 0:
                matrix = np.dot(matrix, rotation_matrix_3d(theta, axis))
        return cls(matrix)


class SequenceMapping(object):

    def __init__(self, mappings: List[ARRAY_MAPPING]) -> None:
        self.mappings = mappings
    
    def __call__(self, array: np.ndarray) -> np.ndarray:
        for mapping in self.mappings:
            array = mapping(array)
        return array


class CartesianSphereMapping(object):

    def __init__(self, r: np.float64) -> None:
        self.r = r
        self.r2 = r*r
    
    def __call__(self, array: np.ndarray) -> np.ndarray:
        size = array.shape[-1]
        neg = array < 0
        result = np.square(array)
        for i in range(1, size):
            result[:, i] = (self.r2 - result[:, :i].sum(axis=1)) * result[:, i]/self.r2
        
        result = np.sqrt(result)
        result[neg] *= -1
        return result



def polar_spherical_mapping(arrays: np.ndarray) -> np.ndarray:
    _, size = arrays.shape
    result = np.zeros(arrays.shape)
    for i in range(size):
        result[:, i] = arrays[:, 0]
    sinc = np.sin(arrays[:, 1:]).cumprod(axis=1)
    coss = np.cos(arrays[:, 1:])

    for index in range(size-1):
        result[:, index] *= sinc[:, -1-index]
        result[:, index+1] *= coss[:, -1-index]
    
    return result


class NoiceMapping(object):

    def __init__(self, distribution: DISTRIBUTION_FUNC) -> None:
        self.distribution = distribution
    
    def __call__(self, arrays: np.ndarray) -> np.ndarray:
        count, d = arrays.shape
        noice = np.empty(arrays.shape, np.float64)
        for i in range(d):
            arrays[:, i] += self.distribution(count)
        
        return arrays
