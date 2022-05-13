from os import stat
from typing import Iterable, List, Dict, Optional, Tuple
from itertools import combinations
import random
import json
from numpy.random.mtrand import random_sample
import pandas as pd


class FrequentItemsetGenerator(object):

    def __init__(
        self, 
        items: Optional[Iterable]=None,
        frequency_range: Tuple[float, float] = (0.1, 0.2),
        union_limits: Tuple[float, float] = (0.2, 0.3),
        isolated_frequency_limit: float = 0.1
        ) -> None:
        """初始化生成器

        :param items: 生成数据的元素集合, defaults to None, 缺省会使用[0, 1, 2... 99]代替。
        :type items: Optional[Iterable], optional
        :param frequency_range: 频繁项集概率分布范围, defaults to (0.1, 0.2)
        :type frequency_range: Tuple[float, float], optional
        :param union_limits: 频繁项并集概率分布限制范围, defaults to (0.2, 0.3)
        :type union_limits: Tuple[float, float], optional
        :param isolated_frequency_limit: 独立项概率分布限制, defaults to 0.1
        :type isolated_frequency_limit: float, optional

        在默认情况下，初始化的元素为：
        
        >>> items = {0, 1, 2, ... 99}

        frequency_range = (0.1, 0.2), 即

        >>> 0.1 < P({0, 1, 2}) < 0.2

        union_limits=(0.2, 0.3), 即

        >>> 0.2 < P({0, 1, 2} | {0, 1} | {0, 2} | {1, 2} | {0} | {1} | {2}) < 0.3

        isolated_frequency_limit=0.1 即

        >>> P({99}) < 0.1
        
        这里 {99} 为独立的一项
        """
        self.items = list(items) if isinstance(items, Iterable) else list(range(100))
        self.frequency_range = frequency_range
        self.union_limits = union_limits
        self.isolated_frequency_limit = isolated_frequency_limit
    
    def proportion_batches(self, itemset_sizes: Iterable) -> list:
        items = self.items.copy()
        random.shuffle(items)
        index = 0
        batches = []
        for itemset_counts in itemset_sizes:
            itemsets = items[index:index+itemset_counts]
            batch = self.itemsets_frequency(itemsets)
            batches.append(list(batch))
            index += itemset_counts

        for item in items[index:]:
            prop = self.random_range(0, self.isolated_frequency_limit)
            batches.append([((item,), prop)])
        
        return batches

    @staticmethod
    def batch2dict(batches: list) -> dict:
        dct = {}
        for batch in batches:
            for key, value in batch:
                dct[key] = value
        return dct
    
    @staticmethod
    def batch2json(batches: list) -> list:
        result = []
        for batch in batches:
            for key, value in batch:
                result.append({
                    "itemset": list(key),
                    "proportion": value
                })
        return json.dumps(result)
    
    @staticmethod
    def batch2df(batches: list) -> pd.DataFrame:
        result = []
        for index, batch in enumerate(batches):
            for key, value in batch:
                result.append({
                    "itemset": list(key),
                    "support": value,
                    "batch": index
                })
        return pd.DataFrame(result)
    
    @staticmethod
    def df2batch(batches: pd.DataFrame) -> list:
        result = []
        for _id, df in batches.groupby("batch"):
            result.append(
                [(tuple(doc["itemset"]), doc["proportion"]) for doc in df.to_dict("record")]
            )
        return result


    @staticmethod
    def write_batch_parquet(filename: str, batches: pd.DataFrame):
        from pyarrow import parquet as pq
        from pyarrow import Table

        table = Table.from_pandas(batches)
        pq.write_table(table, filename)
    
    @staticmethod
    def read_batch_parquet(filename: str) -> pd.DataFrame:
        from pyarrow import parquet as pq
        return pq.read_table(filename).to_pandas()

    def generate_from_batch(self, batches: list, samples: int):
        data = [list() for i in range(samples)]
        for batch in batches:
            itemset_counts = [(itemset, int(prop*samples)) for itemset, prop in batch]
            total_counts = sum([item[1] for item in itemset_counts])
            sample_itemsets = random.sample(data, total_counts)
            index = 0
            for itemset, count in itemset_counts:
                for sample in sample_itemsets[index:index+count]:
                    sample.extend(itemset)
                index += count
        return data

    def generate(self, itemset_sizes: Iterable=(3,)*5, samples: int=100000) -> Tuple[List, Dict]:
        batches = self.proportion_batches(itemset_sizes)
        data = self.generate_from_batch(batches, samples)
        return data, self.batch2dict(batches)

    def generate_df(self, itemset_sizes: Iterable=(3,)*5, samples: int = 100000):
        """生成样本数据，同generate

        :return: (data, proportion)
            samples: 生成的数据样本
            proportion: 数据样本中各项的比例 
        :rtype: Tuple[pandas.DataFrame, Dict]
        """
        data, proportions = self.generate(itemset_sizes, samples)
        return pd.DataFrame({"itemsets": data}), proportions

    @staticmethod
    def gen_empty_list(size: int):
        for i in range(size):
            yield list()
        
    @staticmethod
    def random_range(start: float, end: float) -> float:
        return start + (end-start) * random.random()

    @staticmethod
    def random_sequence(start: float, end: float, size: int):
        block_size = (end-start)/size
        return [FrequentItemsetGenerator.random_range(start+i*block_size, start+(i+1)*block_size) for i in range(size)]

    def itemsets_frequency(self, itemsets: List):
        combs = []
        for size in reversed(range(1, len(itemsets)+1)):
            for comb in combinations(itemsets, size):
                combs.append(comb)
        sup = self.random_range(self.frequency_range[0], self.frequency_range[1])
        limit = self.random_range(self.union_limits[0], self.union_limits[1])
        acc_frequency = self.random_sequence(sup, limit, len(combs))
        frequency = acc_frequency.copy()
        for i, f in enumerate(map(lambda a,b: a-b, acc_frequency[1:],acc_frequency[:-1])):
            frequency[i+1] = f

        return zip(combs, frequency)
