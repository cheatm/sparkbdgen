from typing import Callable, List
import numpy as np
from .types import DISTRIBUTION_FUNC, ARRAY_MAPPING


__all__ = ["ClusterBase", "SingleCluster", "CombinataryCluster"]

class ClusterBase(object):

    def generate(self, count: int) -> np.ndarray:
        raise NotImplementedError()

    def dimension(self) -> int:
        raise NotImplementedError()


class SingleCluster(ClusterBase):

    def __init__(self, distributions: List[DISTRIBUTION_FUNC], mapping: ARRAY_MAPPING=None) -> None:
        super().__init__()
        self.distributions = distributions
        self.mapping = mapping

    def dimension(self) -> int:
        return len(self.distributions)

    def generate(self, count: int) -> np.ndarray:
        data = np.ndarray((len(self.distributions),count), np.float64)
        for index, distribution in enumerate(self.distributions):
            data[index] = distribution(count)
        data = data.T

        if self.mapping is not None:
            data = self.mapping(data)
        return data
    

class CombinataryCluster(ClusterBase):

    def __init__(
        self, 
        items: List[ClusterBase], 
        shifts: np.ndarray, 
        proportions: List[float]=None,
        mapping: Callable[[np.ndarray], np.ndarray]=None
    ) -> None:
        super().__init__()
        assert shifts.shape[0] == len(items)
        dimension = shifts.shape[1]
        for cluster in items:
            assert cluster.dimension() == dimension
        if isinstance(proportions, list):
            assert len(proportions) == len(items)
        else:
            proportions = [1.0] * len(items)
        self._dimension = dimension
        self.items = items
        self.shifts = shifts
        self.proportions = proportions
        self.mapping = mapping
        
    def dimension(self) -> int:
        return self._dimension

    def generate(self, count: int) -> np.ndarray:
        s = sum(self.proportions)
        counts = [int(count*p/s) for p in self.proportions]
        counts[-1] += count - sum(counts)
        result = np.ndarray((count, self.dimension()), np.float64)
        start = 0
        for index, number in enumerate(counts):
            cluster = self.items[index]
            shift = self.shifts[index]
            result[start:start+number] = cluster.generate(number) + shift
            start = start + number
        
        if self.mapping is not None:
            for index, array in enumerate(result):
                result[index] = self.mapping(array)
        
        return result
