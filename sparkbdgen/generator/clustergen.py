from typing import Any, Callable, List, Tuple, Type
import typing
import numpy as np
from .types import DISTRIBUTION_FUNC, ARRAY_MAPPING


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

class ClustersGenerator():

    def __init__(self, dimension: int, clusters: List[ClusterBase]) -> None:
        self.dimension = dimension
        for cluster in clusters:
            assert cluster.dimension() == self.dimension, f"dimension: {cluster.dimension()}, {self.dimension}"
        self.clusters = clusters

    def generate(self, counts: List[int], shifts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert shifts.shape[1] == self.dimension
        assert len(counts) == len(self.clusters) == shifts.shape[0]
        cluster_datas = []
        cluster_labels = []
        for label in range(len(counts)):
            count = counts[label]
            cluster = self.clusters[label]
            shift = shifts[label]
            data = cluster.generate(count)
            data += shift
            cluster_datas.append(data)
            cluster_labels.append(np.array([label]*count))
        features = np.concatenate(cluster_datas)
        labels = np.concatenate(cluster_labels)
        return features, labels 
