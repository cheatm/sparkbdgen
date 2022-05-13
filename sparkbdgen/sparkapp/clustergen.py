from typing import List
from pyspark.sql.session import SparkSession
from sparkbdgen.clusters import ClustersGenerator



def distributed_gen(spark: SparkSession, generators: List[ClustersGenerator], counts: List[int]):
    pass