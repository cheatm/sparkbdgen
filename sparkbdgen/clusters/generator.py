from .clustergen import ClusterBase
from .types import CLUSTER_SHIFT
import numpy as np

from typing import List, Tuple


__all__ = ["ClustersGenerator"]


class ClustersGenerator():

    def __init__(self, dimension: int, clusters: List[ClusterBase], shifter: CLUSTER_SHIFT) -> None:
        self.dimension = dimension
        for cluster in clusters:
            assert cluster.dimension() == self.dimension, f"dimension: {cluster.dimension()}, {self.dimension}"

        self.clusters = clusters
        self.shifter = shifter

    def generate(self, counts: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        assert len(counts) == len(self.clusters)
        cluster_datas = []
        cluster_labels = []
        shifts = self.shifter(self.dimension, len(self.clusters))
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
