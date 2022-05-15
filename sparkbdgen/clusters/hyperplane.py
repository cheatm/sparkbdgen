import numpy as np
from typing import List


class Hyperplane(object):

    def __init__(self, vector: np.ndarray, d: np.float64):
        self._vector = vector
        self._l2 = np.sqrt(np.sum(vector*vector))
        self._d = d
    
    @property
    def dimension(self) -> int:
        return len(self._vector)
    
    def distance(self, target: np.ndarray) -> np.float64:
        shape = target.shape
        if len(shape) == 1:
            assert shape[0] == len(self._vector), "dimension not match"
            return np.abs(self._d + np.dot(target, self._vector))/self._l2
        else:
            raise ValueError(f"Invalid input len(shape): {len(shape)}")
    
    def distances(self, targets: np.ndarray) -> np.ndarray:
        shape = targets.shape
        if len(shape) == 2:
            assert shape[1] == len(self._vector), "dimension not match"
            return np.abs(self._d + np.dot(targets, self._vector))/self._l2
        else:
            raise ValueError(f"Invalid input len(shape): {len(shape)}")

    def side(self, target: np.ndarray) -> bool:
        shape = target.shape
        if len(shape) == 1:
            assert shape[0] == len(self._vector), "dimension not match"
            return self._d + np.dot(target, self._vector) > 0
        else:
            raise ValueError(f"Invalid input len(shape): {len(shape)}")

    def sides(self, target: np.ndarray) -> np.ndarray:
        shape = target.shape
        if len(shape) == 2:
            assert shape[1] == len(self._vector), "dimension not match"
            return self._d + np.dot(target, self._vector) > 0
        else:
            raise ValueError(f"Invalid input len(shape): {len(shape)}")


class HyperplaneSplitter(object):

    def __init__(self, hyperplanes: List[Hyperplane]) -> None:
        self.hyperplanes = hyperplanes

    def __call__(self, data: np.ndarray) -> np.ndarray:
        slabels = np.zeros(len(data), np.int32)
        for hp in self.hyperplanes:
            sides = hp.sides(data)
            slabels = slabels*2 + sides
        return slabels

