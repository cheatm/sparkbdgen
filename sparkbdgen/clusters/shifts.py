from functools import partial
import numpy as np
import math
from typing import List, Tuple, Callable
from .types import CLUSTER_SHIFT
from itertools import product


def generate_random_vector(ranges: List[Tuple[float, float]]) -> List[float]:
    """在指定范围内按均匀分布生成随机向量

    :param ranges: 每个维度的分布范围
    :type ranges: List[Tuple[float, float]]
    :return: [description]
    :rtype: List[float]
    """
    return [(end-begin) * np.random.random() + begin for begin, end in ranges]


def iter_blocks(n: int, fragments: List[List[Tuple[float, float]]]):
    """按切片迭代空间分块

    :param n: 迭代数
    :type n: int
    :param fragments: List[空间分块]
    :type fragments: List[List[Tuple[float, float]]]
    
    :yield: 空间分布范围
    :rtype: Iterator[List[Tuple[float, float]]]
    """
    iterator = product(*fragments)
    for i in range(n):
        yield next(iterator)
    

def ramdom_vectors_by_blocks(n: int, fragments: List[List[Tuple[float, float]]]) -> List[List[float]]:
    """按空间分块均匀分布生成随机向量

    :param n: 向量数
    :type n: int
    :param fragments: List[空间分块]
    :type fragments: List[List[Tuple[float, float]]]
    :return: List[随机向量]
    :rtype: List[List[float]]
    """
    return [generate_random_vector(block) for block in iter_blocks(n, fragments)]


def random_vectors_by_partition(n: int, dimension: int)  -> List[List[float]]:
    """生成随机向量组

    :param n: 生成向量数
    :type n: int
    :param dimension: 向量维度
    :type dimension: int
    :return: 随机向量组
    :rtype: List[List[float]]

    >>> [
            [0.09866185481764456, 0.24677911604333763, 0.2564142446564428, 0.3661132487605418]
            [0.1959593783451019, 0.22758231911002214, 0.49157089229502204, 0.9325823551379472]
            [0.13523454915437688, 0.2734214436483937, 0.7084943844489697, 0.287845955930896]
            [0.19115298679551979, 0.3999493111669893, 0.5274002205879706, 0.673410868393441]
        ]
    """
    part_counts = math.ceil((math.pow(n, 1/dimension)))
    sep = 1 /part_counts
    fragment_parts = [(i*sep, (i+1)*sep) for i in range(part_counts)]
    fragments = [fragment_parts] * dimension
    return ramdom_vectors_by_blocks(n, fragments)


def grid_centers(n: int, dimension: int, mapping: Callable[[np.ndarray], np.ndarray]=None) -> np.ndarray:
    if mapping is None:
        return np.array(random_vectors_by_partition(n, dimension))
    else:
        # return np.vectorize(mapping)(np.array(random_vectors_by_partition(n, dimension)))
        vectors = [mapping(np.array(v)) for v in random_vectors_by_partition(n, dimension)]

        return np.array(vectors)

    


def grid(mapping: Callable[[np.ndarray], np.ndarray]=None) -> CLUSTER_SHIFT:
    return partial(grid_centers, mapping=mapping)