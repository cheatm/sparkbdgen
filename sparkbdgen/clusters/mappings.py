from typing import List
import numpy as np
from .types import *


__all__ = [
    "LinearMapping", "CartesianSphereMapping", "NoiceMapping",
    "SequenceMapping", 
    "polar_spherical_mapping"
]


def rotation_matrix_3d(theta: np.float64, axis: int) -> np.ndarray:
    sint = np.sin(theta)
    cost = np.cos(theta)
    result = np.identity(3)
    result[(axis+1)%3, (axis+1)%3] = cost
    result[(axis+2)%3, (axis+1)%3] = sint
    result[(axis+1)%3, (axis+2)%3] = -sint
    result[(axis+2)%3, (axis+2)%3] = cost
    return result


def normalize(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr)
    if norm > 0:
        return arr/norm
    return arr

def rodrigues(axis: np.ndarray, theta: np.float64):
    assert len(axis) == 3
    arr = normalize(axis)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return cos*np.identity(3) + (1-cos)*np.outer(arr, arr) + sin*np.array([
        [0, -arr[2], arr[1]],
        [arr[2], 0, -arr[0]],
        [-arr[1], arr[0], 0],
    ], np.float64)


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

    def rotation2D(self, theta: np.float64):
        assert self.matrix.shape == (2, 2)
        sint = np.sin(theta)
        cost = np.cos(theta)
        matrix = np.array([
            [cost,-sint],
            [sint,cost]
        ])
        return LinearMapping(np.dot(matrix, self.matrix))

    @classmethod
    def Rotation2D(cls, theta: np.float64):
        sint = np.sin(theta)
        cost = np.cos(theta)
        return cls(np.array([
            [cost,-sint],
            [sint,cost]
        ]))

    @classmethod
    def Rotation3D(cls, roll: np.float64=0, pitch: np.float64=0, yaw: np.float64=0):
        matrix = np.identity(3, np.float64)
        for axis, theta in enumerate([roll, pitch, yaw]):
            if theta != 0:
                matrix = np.dot(rotation_matrix_3d(theta, axis), matrix)
        return cls(matrix)

    def rodrigues(self, axis: np.ndarray, theta: np.float64):
        assert self.matrix.shape == (3, 3)
        matrix = rodrigues(axis, theta)
        return LinearMapping(np.dot(matrix, self.matrix))

    @classmethod
    def Rodrigues(cls, axis: np.ndarray, theta: np.float64):
        return cls(rodrigues(axis, theta))


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
