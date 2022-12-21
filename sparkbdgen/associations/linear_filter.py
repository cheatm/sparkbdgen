from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd


def basket(supports: list, limit: float=1e-4) -> np.ndarray:
    """generate independent supports

    :param supports: supports of single item
    :type supports: list
    :param limit: _description_, defaults to 1e-4
    :type limit: float, optional
    :return: supports of combinatary items
    :rtype: np.ndarray
    """
    l = len(supports)
    t = 1 << len(supports)
    batches = np.zeros(t, float)
    batches[0] = 1
    index = list(range(l))
    for size in range(1, l+1):
        for items in combinations(index, size):
            pos = 0
            sp = 1
            for i in items:
                pos += 1 << i
                sp *= supports[i]
                if sp >= limit:
                    batches[pos] = sp

    return batches


def midsolve(matrix: np.ndarray, begin: int, l: int):
    """Solve combination matrix

    :param matrix: supports of combinatary items or combination matrix
    :type matrix: np.ndarray
    :param begin: begin index
    :type begin: int
    :param l: itemset size
    :type l: int

    :requile: 2^l = length of matrix 
    """
    if l == 0:
        return 

    l -= 1
    size = 1 << l
    mid = begin + size
    end = mid+ size

    matrix[begin:mid] -= matrix[mid:end]
    midsolve(matrix, begin, l)
    midsolve(matrix, mid, l)


def combination_matrix(l: int) -> np.ndarray:
    """generate matrix like:
    >>> [[A, A]
         [0, A]]

    >>> when l = 3:
        [[1 1 1 1 1 1 1 1]
         [0 1 0 1 0 1 0 1]
         [0 0 1 1 0 0 1 1]
         [0 0 0 1 0 0 0 1]
         [0 0 0 0 1 1 1 1]
         [0 0 0 0 0 1 0 1]
         [0 0 0 0 0 0 1 1]
         [0 0 0 0 0 0 0 1]]

    :param l: item count of combination
    :type l: int
    :return: 
    :rtype: np.ndarray
    """
    size = 1 << l
    M = np.zeros((size, size), int)
    M[0][0] = 1
    M[0][1] = 1
    M[1][1] = 1
    for i in range(1, l):
        s = 1 << i
        e = s << 1
        M[:s, s:e] = M[:s, :s]
        M[s:e, s:e] = M[:s, :s]
    return M


def combination_matrix_inv(l: int) -> np.ndarray:
    """generate inversed matrix of combinatian matrix like:
    >>> [[A, -A]
         [0,  A]]

    >>> when l = 3:
        [[ 1 -1 -1  1 -1  1  1 -1]
         [ 0  1  0 -1  0 -1  0  1]
         [ 0  0  1 -1  0  0 -1  1]
         [ 0  0  0  1  0  0  0 -1]
         [ 0  0  0  0  1 -1 -1  1]
         [ 0  0  0  0  0  1  0 -1]
         [ 0  0  0  0  0  0  1 -1]
         [ 0  0  0  0  0  0  0  1]]

    :param l: item count of combination
    :type l: int
    :return: _description_
    :rtype: np.ndarray
    """
    size = 1 << l
    M = np.zeros((size, size), int)
    M[0][0] = 1
    M[0][1] = -1
    M[1][1] = 1
    for i in range(1, l):
        s = 1 << i
        e = s << 1
        M[:s, s:e] = -M[:s, :s]
        M[s:e, s:e] = M[:s, :s]
    return M


def disjoint(dsets: Dict[int, set], a: int, b: int):
    """disjoin a & b

    :param dsets: disjoinset pointer
    :type dsets: Dict[int, set]
    :param a: item a
    :type a: int
    :param b: item b
    :type b: int
    """
    A = a in dsets
    B = b in dsets
    if A and B:
        if id(dsets[a]) != id(dsets[b]):
            aset = dsets[a]
            bset = dsets[b]

            for i in bset:
                dsets[i] = aset
            aset.update(bset)
            del bset
    elif A:
        dsets[a].add(b)
        dsets[b] = dsets[a]
    elif B:
        dsets[b].add(a)
        dsets[a] = dsets[b]
    else:
        s = set((a, b))
        dsets[a] = s
        dsets[b] = s
    

def disjointset(itemlist: List[List[int]]) -> Dict[int, set] :
    """find disjoint sets from itemsets

    :param itemlist: list of itemsets
    :type itemlist: List[List[int]]
    :return: disjoint sets and item mapping
    :rtype: Dict[int, set]
    """
    dsets: Dict[int, set] = {}

    for items in itemlist:
        if len(items) == 1:
            if items[0] not in dsets:
                dsets[items[0]] = set(items)
        else:
            for i in range(len(items)-1):
                disjoint(dsets, items[i], items[i+1])
    return dsets


@dataclass
class Batch:

    freqent: List[int]
    items: List[Tuple[List[int], float]]
    
    def __post_init__(self):
        self.freqent.sort()
        self.itempos = {}
        for i, item in enumerate(self.freqent):
            self.itempos[item] = i
        self.L = len(self.freqent)
        self.N = 1 << self.L

    def bisupports(self) -> np.ndarray:
        l = len(self.freqent)
        size = 1 << l
        sarray = np.zeros((size,), float)
        sarray[0] = 1
        for items, support in self.items:
            sarray[self.biindex(items)] = support

        return sarray
    
    def biindex(self, items: List[int]):
        index = 0
        for item in items:
            index += 1 << self.itempos[item]
        return index


def require(condition: bool, msg: str):
    if not condition:
        raise ValueError(f"Require: {msg}")


def basic_solution(l: int, A: np.ndarray, Y: np.ndarray, frees: list, solids: list) -> np.ndarray:
    n, m = A.shape
    require(n == len(Y), "len(A) = len(Y)")
    require(len(frees)+len(solids) == m, "len(frees) + len(solids) = width(A)")

    INV = combination_matrix_inv(l)[:, solids]
    X = INV.dot(Y)[solids]
    freeX: np.ndarray = None
        
    vcount = len(frees)
    # find valid vertex
    Sx = (INV.dot(A)[solids][:, frees] * (-1))
    for items in rcombinations(m, n):
        B = A[:, items]
        if np.linalg.matrix_rank(B) != n:
            continue
        Xb = np.linalg.inv(B).dot(Y)
        if xavailable(Xb):
            if freeX is None:
                freeX = get_free_vector(Xb, items, frees, m)
            else:
                freeX = randomVector(freeX, get_free_vector(Xb, items, frees, m))
            vcount -= 1
            if vcount == 0:
                break
    
    require(freeX is not None, "No vertex found")
    print(freeX)
    S = Sx.dot(freeX) + X
    R = np.zeros((m, ), float)

    for i, s in enumerate(solids):
        R[s] = S[i]

    for i, f in enumerate(frees):
        R[f] = freeX[i]

    return R


def align_index(xindex: list, vindex: list):
    l = 0
    x = 0
    v = 0
    xtarget = []
    vtarget = []
    while x < len(xindex) and v < len(vindex):
        if xindex[x] < vindex[v]:
            x += 1
        elif xindex[x] > vindex[v]:
            v += 1
        else:
            xtarget.append(x)
            vtarget.append(v)
            l += 1
            x += 1
            v += 1

    return l, xtarget, vtarget


def rcombinations(m: int, n: int):
    for items in combinations(reversed(range(m)), n):
        yield list(reversed(items))


def smerge(r: list, l1: list, l2: list):
    i = 0
    i1 = 0
    i2 = 0
    while i1 < len(l1) and i2 < len(l2):
        if l1[i1] < l2[i2]:
            r[i] = l1[i1]
            i1 += 1
        else:
            r[i] = l2[i2]
            i2 += 1
        i += 1

    while i1 < len(l1):
        r[i] = l1[i1]
        i1 += 1
        i += 1
    
    while i2 < len(l2):
        r[i] = l2[i2]
        i2 += 1
        i += 1


def scombinations(parents: list, subs: list):
    l = len(parents)
    yield list(parents)
    for s in range(1, l):
        if s > len(subs):
            break
        for pitems in combinations(parents, l - s):
            for sitems in combinations(reversed(subs), s):
                sitems = list(reversed(sitems))
                comb = [0] * l
                smerge(comb, pitems, sitems)
                yield comb


def search_available(A: np.ndarray, Y: np.ndarray, xindex: list, vindex: list):
    n, m = A.shape
    lv = len(vindex)
    freeX: np.ndarray = None
    rcount = len(vindex)

    searched = 0
    # for items in combinations(range(m), n):
    # for items in rcombinations(m, n):
    for items in scombinations(vindex, xindex):
        searched += 1        
        lenght, xtargets, vtargets = align_index(items, vindex)
        if lenght == 0:
            continue
        B = A[:, items]
        if np.linalg.matrix_rank(B) != n:
            continue
        Xb = np.linalg.inv(B).dot(Y)
        if xavailable(Xb):
            vertex = np.zeros((lv,), float)
            vertex[vtargets] = Xb[xtargets]
            
            print("searched:", searched)
            print(items, vtargets, vertex)
            if freeX is None:
                freeX = vertex
            else:
                freeX = randomVector(freeX, vertex, lambda: 0.5)
                rcount -= 1
                if rcount == 0:
                    break
    return freeX


def get_free_vector(X: np.ndarray, items: list, frees: list, m: int):
    T = np.zeros((m,))
    for i in range(len(items)):
        T[items[i]] = X[i]
    return T[frees]


def completeX(X: np.ndarray, index: Iterable, n: int):
    R = np.zeros(n)
    for i, x in enumerate(index):
        R[x] = X[i]
    return R


def randomVector(source: np.ndarray, target: np.ndarray, random_gen=np.random.rand):
    vector = target - source
    return source + random_gen() * vector


def xavailable(X: np.ndarray):
    for x in X:
        if x < 0:
            return False
    return True  



def freeindex(Y: np.ndarray):
    r = []
    for i, y in enumerate(Y):
        if y == 0:
            r.append(i)
    return r


def test_bascket():
    batches = basket([0.5, 0.5, 0.5])
    print(batches)
    midsolve(batches, 0, 3)
    print(batches)
    

def test_matrix(l: int=3):

    M = combination_matrix(l)
    print(M)
    MI = combination_matrix_inv(l)
    print(MI)


def test_disjoint_set():
    pass


def expand_matrix(l: int, freeindex: list):
    EX = np.zeros((1 << l, len(freeindex)), int)
    for i, n in enumerate(freeindex):
        EX[n, i] = 1 
    return EX


def test_expand(batch: Batch):
    l = batch.L
    Y = batch.bisupports()

    n = 1 << l
    M = combination_matrix(l)
    MI = combination_matrix_inv(l)
    freenums = freeindex(Y)
    Y[freenums] = 0.1

    EX = expand_matrix(l, freenums)
    print(np.concatenate([M, EX], axis=1))
    print(MI.dot(EX))
    df = pd.DataFrame(MI.dot(EX))
    df["Y"] = Y

    print(df)


def limit_matrix(l: int, Y: np.ndarray, limit: float=0.1):
    frees = []
    Y = Y.copy()
    
    for i, y in enumerate(Y):
        if not (y > 0):
            Y[i] = limit
            frees.append(i)
    
    EX = np.zeros((len(Y), len(frees)), int)
    for i, f, in enumerate(frees):
        EX[f, i] = 1

    size = 1 << l
    M = combination_matrix(l)

    return np.concatenate([M, EX], axis=1), Y, list(range(size)), list(range(size, size+len(frees)))
    

def coefficient_array(n: int, l: int):
    size = 1 << l
    array = np.zeros((size,))
    array[n] = 1
    fill_co_array(array, n, 0, l)


def fill_co_array(array: np.ndarray, n: int, start: int, l: int):
    if l == 0:
        return
    l -= 1
    size = 1 << l
    mid = start+size
    if n < mid:
        fill_co_array(array, n, start, l)
        array[mid:mid+size] = array[start:mid]
    elif n > mid:
        fill_co_array(array, n, mid, l)
    else:
        end = mid + size
        start = mid
        size = 1
        mid = start + size
        while mid + size <= end:
            array[mid:mid+size] = array[start:mid]
            size = size << 1
            mid = start + size


def fill_co_inv_array(array: np.ndarray, n: int, start: int, l: int):
    if l == 0:
        return
    l -= 1
    size = 1 << l
    mid = start+size
    if n < mid:
        fill_co_inv_array(array, n, start, l)
        array[mid:mid+size] = -array[start:mid]
    elif n > mid:
        fill_co_inv_array(array, n, mid, l)
    else:
        end = mid + size
        start = mid
        size = 1
        mid = start + size
        while mid + size <= end:
            array[mid:mid+size] = -array[start:mid]
            size = size << 1
            mid = start + size


def test_basic_solution(batch: Batch):

    Y = batch.bisupports()

    # frees = []
    # solids = []
    # for i, y in enumerate(Y):
    #     if y > 0:
    #         solids.append(i)
    #     else:
    #         frees.append(i)
    # M = combination_matrix(batch.L)

    M, Y, solids, frees = limit_matrix(batch.L, Y, 0.2)

    R = basic_solution(batch.L, M[solids], Y[solids], frees, solids)
    df = pd.DataFrame(M)
    X = R[:1<<batch.L]
    df["X"] = X
    df["Y"] = M[:, :1<<batch.L].dot(X)
    print(df)
    

def test_co_array(l: int=3):
    size = 1 << l
    A = np.zeros((size, size), int)
    INVA = np.zeros((size, size), int)
    for n in range(size):
        # A[n][n] = 1
        # fill_co_array(A[n], n, 0, l)
        INVA[n][n] = 1
        fill_co_inv_array(INVA[n], n, 0, l)

    # print(A)
    print(INVA)


def test_basic():

    # Y = basket([0.5, 0.5, 0.5])
    batch = Batch(
        [0, 1, 2, 3],
        [
                # ([0, 1, 2], 0.1),
            ([0, 1], 0.25),
            ([1, 2], 0.25),
            ([2, 3], 0.25),
            # ([0, 3], 0.25),
            ([0], 0.5),
            ([1], 0.5),
            ([2], 0.5),
            ([3], 0.5),
        ]
    )
    Y = batch.bisupports()
    A = combination_matrix(4)
    vindex = []
    yindex = []
    for i, y in enumerate(Y):
        if y == 0:
            vindex.append(i)
        else:
            yindex.append(i)
    # print(Y)
    # print(vindex)
    vertex = search_available(A[yindex], Y[yindex], yindex, vindex)
    print(vertex)


def test_comb():

    # for items in scombinations([0, 1, 2], [4, 5, 6]):
    for items in rcombinations(4, 2):
        print(items)



def main():
    # test_bascket()
    # test_matrix()

    batch = Batch(
        [0, 1, 2, 3],
        [
                # ([0, 1, 2], 0.1),
            ([0, 1], 0.25),
            ([1, 2], 0.25),
            ([2, 3], 0.25),
            # ([0, 3], 0.25),
            ([0], 0.5),
            ([1], 0.5),
            ([2], 0.5),
            ([3], 0.5),
        ]
    )
    # test_expand(batch)

    # test_basic_solution(batch)
    # test_co_array()
    test_basic()
    # test_comb()


if __name__ == "__main__":
    main()