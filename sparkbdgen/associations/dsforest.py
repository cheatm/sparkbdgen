from typing import List, Tuple
from dataclasses import dataclass


def union(dsf: list, a: int, b: int):
    ra = root(dsf, a)
    rb = root(dsf, b)

    dsf[rb] = ra


def root(dsf: list, a: int):
    r = dsf[a]

    if r >= 0:
        if dsf[r] == r:
            return r
        else:
            r = root(dsf, r)
            dsf[a] = r
            return r
    return r        


def disjointset_forest(n: int, itemsets: List[List[int]]):
    dsf = [-1] * n

    for items in itemsets:
        if dsf[items[0]] == -1:
            dsf[items[0]] = items[0]
        
        a = items[0]
        for i in range(1, len(items)):
            b = items[i]
            r = root(dsf, b)
            if r == -1:
                dsf[b] = dsf[a]
            else:
                if root(dsf, a) != r:
                    union(dsf, a, b)

            a = b
    
    return dsf



def disjoint_batch(n: int, itemsets: List[Tuple[List[int], float]]):
    dsf = disjointset_forest(n, [items for items, _ in itemsets])
    
    dsets = {}
    for i in range(n):
        r = root(dsf, i)

        if r in dsets:
            dsets[r][0].append(i)
        elif r >= 0:
            dsets[r] = ([i], [])


    for items, support in itemsets:
        key = root(dsf, items[0])
        dsets[key][1].append((items, support))

    return dsets


def test_dsf():
    itemsets = [
        [1, 2, 3],
        [1, 2],
        [2, 3],
        [4],
        [5],
        [4, 5, 6],
    ]

    dsf = disjointset_forest(7, itemsets)
    print(dsf)


def test_batch():
    itemsets = [
        ([1, 2, 3], 0.1),
        ([1, 2], 0.2),
        ([2, 3], 0.2),
        ([4], 0.4),
        ([5], 0.5),
        ([4, 5, 6], 0.1),
    ]

    r = disjoint_batch(7, itemsets)
    print(r)


def main():
    # test_dsf()
    test_batch()


if __name__ == "__main__":
    main()