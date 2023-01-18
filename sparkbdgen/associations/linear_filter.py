from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd

pd.set_option("precision", 4)


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


def solveY(Y: np.ndarray, l: int):
    X = Y.copy()
    midsolve(X, 0, l)
    return X


def solve_full_rank(Y: np.ndarray, l: int):
    X = Y.copy()
    midsolve(X, 0, l)
    return X


def solution_matrix(l: int, xindex: list, yindex: list):
    A = combination_matrix(l)
    A[xindex, :] = 0
    midsolve(A, 0, l)
    return A[yindex][:,xindex]


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
    S = Sx.dot(freeX) + X
    R = np.zeros((m, ), float)

    for i, s in enumerate(solids):
        R[s] = S[i]

    for i, f in enumerate(frees):
        R[f] = freeX[i]

    return R


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


def align_index(xindex: list, independents: list):
    l = 0
    x = 0
    v = 0
    xtarget = []
    vtarget = []
    while x < len(xindex) and v < len(independents):
        if xindex[x] < independents[v]:
            x += 1
        elif xindex[x] > independents[v]:
            v += 1
        else:
            xtarget.append(x)
            vtarget.append(v)
            l += 1
            x += 1
            v += 1

    return l, xtarget, vtarget


def get_groups(A: np.ndarray):
    N, M = A.shape
    groups = []
    n = 0
    m = 0
    group = []
    while (n < N):
        if A[n][m] == 1:
            group.append(m)
            n += 1
            m += 1
        else:
            n -= 1
            


def search_available(A: np.ndarray, Y: np.ndarray, independents: list):
    n, m = A.shape
    rcount = len(independents)
    searched = 0
    solutions = np.zeros((rcount, m))
    base_counts = np.zeros((rcount, ), int)
    scount = 0
    ed = set(independents)
    for items in rcombinations(m, n):
        searched += 1        
        B = A[:, items]
        if np.linalg.matrix_rank(B) != n:
            continue
        Xb = np.linalg.inv(B).dot(Y)
        if xavailable(Xb):
            solutions[scount, items] = Xb
            has_new = False
            for k in list(ed):
                if solutions[scount, k] != 0 :
                    ed.discard(k)
                    has_new = True
            
            if has_new:
                base_counts[scount] = searched
                scount += 1
                if len(ed) == 0:
                    break
                if scount == len(solutions):
                    break
            else:
                solutions[scount, items] = 0
    
    vdf = pd.DataFrame(solutions[:scount], index=base_counts[:scount])
    print(vdf[independents])
    for i in range(1, scount):
        solutions[i, independents] = randomVector(solutions[i-1, independents], solutions[i, independents])

    return solutions[scount-1, independents]


def basic_solution_matrix(L: int, dependents: list, indpendents: list):
    A = combination_matrix(L)
    for i in indpendents:
        A[i, :] = 0
    
    midsolve(A, 0, L)
    return A[dependents][:,indpendents]
    

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


def float_random_solution(L: int, Y: np.ndarray):
    dependents = []
    independents = []
    A = combination_matrix(L)
    for i in range(len(Y)):
        if Y[i] == 0:
            independents.append(i)
            A[i, :] = 0
        else:
            dependents.append(i)
    B = solve_full_rank(A, L)[dependents]
    print(pd.DataFrame(B, dependents))
    vector = search_available(A[dependents], Y[dependents], independents)
    BS = solve_full_rank(A, L)[dependents][:, independents]
    BX = solve_full_rank(Y, L)[dependents]
    # DX = ((-1) * BS).dot(vector) + BX
    X = np.zeros((1<<L,), float)
    # X[dependents] = DX
    # X[independents] = vector
    return X


def test_rank():
    L = 3
    size = 1 << L
    A = combination_matrix(L)
    independent = [3,5,6,7]
    dependent = [0, 1, 2, 3, 4, 5]
    # A[independent, :] = 0
    

    R = size
    # A = solve_full_rank(A, 3)
    # print(A)
    A = expand_matrix(L, freeindex=independent)
    print(A)
    F = 0
    N = 0
    for items in combinations(range(size+len(independent)), size):
        B = A[:, list(items)]
        r = np.linalg.matrix_rank(B)
        if r < R:
            N += 1
        else:
            F += 1
            print(B)
            print(items, r)
    print(A)
    print(f"N={N}")
    print(f"F={F}")


def expand_matrix(l: int, freeindex: list):
    A = combination_matrix(l)
    EX = np.zeros((1 << l, len(freeindex)), int)
    for i, n in enumerate(freeindex):
        EX[n, i] = 1 
    return np.concatenate([A, EX], axis=1)


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

    M, Y, solids, frees = limit_matrix(batch.L, Y, 0.2)

    R = basic_solution(batch.L, M[solids], Y[solids], frees, solids)
    df = pd.DataFrame(M)
    X = R[:1<<batch.L]
    df["X"] = X
    df["Y"] = M[:, :1<<batch.L].dot(X)
    print(df)


def sub_index(items: list, L: int):
    size = 1 << L
    others = []
    for l in range(L):
        if l not in items:
            others.append(l)
    root = sum([1 << i for i in items])
    for o in range(len(others)+1):
        for oindex in combinations(others, o):
            yield root + sum([1 << i for i in oindex])    

def random_y(L: int, ns: int=2, chance: float=0.5):

    size = 1 << L
    X = np.random.rand(size)
    X = X / sum(X)
    Y = combination_matrix(L).dot(X)

    for items in combinations(range(L), ns):
        if np.random.rand() < chance:
            for i in sub_index(items, L):
                Y[i] = 0

    return Y


def test_no_full_rank(L: int, chance: float=0.5):

    Y = random_y(L, chance=chance)
    print(Y)
    X = float_random_solution(L, Y)
    # print(pd.DataFrame({
    #     "X": X,
    #     "Y": Y
    # }))


def distance(plane: np.ndarray, d: float, vector: np.ndarray):
    return (d + plane.dot(vector)) / np.linalg.norm(plane)


def pdot(p1: np.ndarray, p2: np.ndarray):
    return (p1/np.linalg.norm(p1)).dot(p2/np.linalg.norm(p2))


def test_distance(Y: np.ndarray, L: int):
    dependents = []
    independents = []

    for i in range(len(Y)):
        if Y[i] == 0:
            independents.append(i)
        else:
            dependents.append(i)
    
    BS = -basic_solution_matrix(L, dependents, independents)
    X = solveY(Y, L)
    ds = X[dependents]
    df = pd.DataFrame(BS)
    df["d"] = ds
    print(df)
    distances = np.zeros(len(dependents), float)
    vector = np.zeros(len(dependents), float)
    for i in range(len(dependents)):
        distances[i] = distance(BS[i], ds[i], vector)
    
    for i in range(1, len(dependents)):
        factor = pdot(BS[0], BS[i])
        tor = -distances[i]/factor
        print(i, factor, tor)



def get_range(F: np.ndarray, S: np.ndarray):
    
    l = 0
    r = 1

    for i in range(len(F)):
        if F[i] == 1:
            l = max(l, -S[i])
            # print(f"{i}: l = {l}")
        elif F[i] == -1:
            r = min(r, S[i])
            # print(f"{i}: r = {r}")
        
    return l, r



def uniform(l: float, r: float):
    return np.random.rand() * (r - l) + l


def y_rand_r(L: int, independents: list, Y: np.ndarray, M: np.ndarray, X:np.ndarray, rfunc=uniform):
    """_summary_

    

    :param L: _description_
    :type L: int
    :param independents: _description_
    :type independents: list
    :param Y: _description_
    :type Y: np.ndarray
    :param M: _description_
    :type M: np.ndarray
    :param X: _description_
    :type X: np.ndarray
    :param rfunc: _description_, defaults to uniform
    :type rfunc: _type_, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    head = f"y_rand_r({L}, {independents}, {Y}, {M}, {X})"
    print(head)
    if L < 1:
        raise ValueError("Require: L >= 1")
    
    if not len(independents):
        print(head, Y)
        return Y

    if L == 1:
        R = np.zeros((2,), float)
        if len(independents) == 2:
            R[0] = rfunc(M[0] + M[1], X[0] + X[1])
        else:
            R[0] = Y[0]
        left = max(M[1], R[0]-X[0])
        right = min(X[1], R[0]-M[0])
        try:
            R[1] = rfunc(left, right)
        except:
            print(head)
            raise
        print(head, R)
        return R
    
    size = 1 << L
    mid = 1 << (L-1)

    i = 0
    while i < len(independents) and independents[i] < mid:
        i += 1

    Y_l = y_rand_r(L-1, independents[:i], Y[:mid], M[:mid]+M[mid:], X[:mid]+X[mid:], rfunc)
    V = combination_matrix_inv(L-1)
    d_y = V.dot(Y_l)
    X_l = d_y - X[:mid]
    print(f"Y_l: {Y_l}")
    print(f"X_l: {X_l}")
    print(f"M_l: {M[:mid]}")
    for _i in range(mid):
        X_l[_i] = max(X_l[_i], M[_i])
    print(f"X_l: {X_l}")

    X_r = d_y - M[:mid]
    print(f"X_r: {X_r}")
    print(f"X  : {X[mid:]}")
    for _i in range(mid):
        X_r[_i] = min(X_r[_i], X[mid+_i])
    
    print(f"X_r: {X_r}")

    
    Y_r = y_rand_r(L-1, [y-mid for y in independents[i:]], Y[mid:], X_l, X_r, rfunc)

    R = np.zeros((size,), float)
    R[:mid] = Y_l
    R[mid:] = Y_r

    print(head, R)

    return R

def y_rand(L: int, independents: list, Y: np.ndarray, rfunc=uniform):
    head = f"y_rand({L}, {independents}, {Y})"
    print(head)
    if L < 1:
        raise ValueError("Require: L >= 1")

    if not len(independents):
        print(head, Y)
        return Y
    if L == 1:
        result = np.zeros((2,), float)
        result[0] = Y[0]
        result[1] = rfunc(0, Y[0])
        print(head, result)
        return result
    
    size = 1 << L
    mid = 1 << (L-1)
    
    i = 0
    while i < len(independents) and independents[i] < mid:
        i += 1

    Y_l = y_rand(L-1, independents[:i], Y[:mid], rfunc)

    V = combination_matrix_inv(L-1)
    X_l = V.dot(Y_l)
    
    Y_r  = y_rand_r(L-1, [y-mid for y in independents[i:]], Y[mid:], np.zeros((mid,), float), X_l, rfunc)
    result = np.zeros((size,), float)
    result[:mid] = Y_l
    result[mid:] = Y_r

    print(head, result)

    return result


def lower(l, r):
    if l > r:
        raise ValueError(f"{l} > {r}")
    
    print("lower", l, r)
    return l


def upper(l, r):
    if l > r:
        raise ValueError(f"{l} > {r}")
    return r


def medium(l, r):
    if l > r:
        raise ValueError(f"{l} > {r}")

    return (l+r) / 2


def yrange(V: np.ndarray, B: np.ndarray, T: np.ndarray, rfunc=uniform):
    l = 0
    r = 1

    for i in range(len(V)):
        if V[i] > 0:
            l = max(l, B[i])
            r = min(r, T[i])
        elif V[i] < 0:
            l = max(l, -T[i])
            r = min(r, -B[i])
    
    return rfunc(l, r)


def vmax(v1: np.ndarray, v2: np.ndarray, size: int):
    v = np.zeros((size,), float)
    for s in range(size):
        v[s] = max(v1[s], v2[s])
    return v


def vmin(v1: np.ndarray, v2: np.ndarray, size: int):
    v = np.zeros((size,), float)
    for s in range(size):
        v[s] = min(v1[s], v2[s])
    return v


def randy(L: int, independents: list, A: np.ndarray, B: np.ndarray, T: np.ndarray, rfunc=uniform):
    if L == 0:
        print(A, B, T)
        return np.array([yrange(A[0], B, T, rfunc)])
    
    rdf = pd.DataFrame(A, columns=independents)
    rdf["B"] = B
    rdf["T"] = T
    print(rdf)

    mid = 1 << (L-1)

    w = 0
    while independents[w] < mid:
        w += 1

    Y = np.zeros((len(independents),), float)
    if w:
        W = A[:mid, :w]
        wy = randy(L-1, independents[:w], W, B[mid:]+B[:mid], T[mid:]+T[:mid], rfunc)
        wx = A[:, :w].dot(wy)
        B = B - wx
        T = T - wx
        Y[:w] = wy

    vy = randy(
        L-1, [pos - mid for pos in independents[w:]], A[mid:, w:], 
        vmax(B[mid:], -T[:mid], mid),
        vmin(T[mid:], -B[:mid], mid),
        rfunc
    )

    Y[w:] = vy

    return Y



def array_match(a1: np.ndarray, a2: np.ndarray, s: int):
    for i in range(s):
        if a1[i] != a2[i]:
            return False
    
    return True


def find_group(V: np.ndarray, pos: int):
    n, m = V.shape

    groups = []
    tag = set()

    for i in range(pos+1):
        if V[i, 0] == 0:
            continue

        if i in tag:
            continue
        group = [i]
        groups.append(group)
        tag.add(i)
        
        for j in range(i+1, pos+1):
            if array_match(V[i, :], V[j, :], m):
                group.append(j)
                tag.add(j)

    return groups


def zero_index(a: np.ndarray, pos: int):
    n = len(a)

    index = []
    for i in range(pos):
        if a[i] == 0:
            index.append(i)
    
    for i in range(pos+1, n):
        index.append(i)

    return index


def find_opposite(V: np.ndarray, index: list, array: np.ndarray):
    _, m = V.shape
    oa = -array
    results = []
    for i in index:
        if array_match(V[i, :], oa, m):
            results.append(i)
    return results


def get_y_range(L: int, V: np.ndarray, X: np.ndarray, index: list):
    require(len(index) > 0, "index size should > 0")
    if len(index) == 1:
        l = 0
        r = 1
        for i in range(index[0]+1):
            if V[i, 0] > 0:
                l = max(l, -X[0])
            elif V[i, 0] < 0:
                r = min(r, X[0])
        return l, r
    
    rdf = pd.DataFrame(V, columns=index)
    rdf["X"] = X
    print(rdf)

    pos = index[0]
    groups = find_group(V, pos)
    zeros = zero_index(V[:, 0], pos)
    print(pos, groups)
    print("zeros", zeros)

    l = 0 
    r = 1
    for group in groups:
        p = group[0]
        opposite = find_opposite(V[:, 1:], zeros, V[p, 1:])
        if not len(opposite):
            i = 0
            # mid = 1 << (L-1)
            size = 1 << L
            m = 1
            while m < L:
                groupsize = 1 << (L - m)
                mid = size - groupsize
                if index[0] < mid:
                    i = 0

                    while index[i] < mid:
                        i += 1
                    print(mid, i)
                    nx = X.copy()
                    for _m in range(m):
                        nx[_m:_m+groupsize] += nx[_m+groupsize:_m+groupsize+groupsize]
                    return get_y_range(L, V[:mid, :i], nx[:mid], index[:i])
                m += 1 

            
        if V[p, 0] > 0:
            l = max(- (min(X[group]) + min(X[opposite])), l)
            print(f"l = {l}", group, opposite)
        else:
            r = min(min(X[group]) + min(X[opposite]), r)
            print(f"r = {r}", group, opposite)

    return l, r


def float_y_range(L: int, V: np.ndarray, X: np.ndarray, independents: list, rfunc=uniform):

    # rdf = pd.DataFrame(V, columns=independents)
    # rdf["X"] = X

    # print(rdf)

    Y = np.zeros((len(independents,)), float)
    for i, pos in enumerate(independents):
        # rdf = pd.DataFrame(V[:, i:], columns=independents[i:])
        # rdf["X"] = X

        # print(rdf)
        # groups = find_group(V[:, i:], pos)
        # index = zero_index(V[:, i], pos)
        # print(pos, groups)
        # l = 0
        # r = 1
        # print("zeros", index)
        # for group in groups:
        #     p = group[0]

        #     opposite = find_opposite(V[:, i+1:], index, V[p, i+1:])
        #     if V[p, i] > 0:
        #         if len(opposite):
        #             l = max(- (min(X[group]) + min(X[opposite])), l)
        #         else:
        #             l = max(- min(X[group]), l)
        #     else:
        #         if len(opposite):
        #             r = min(min(X[group]) + min(X[opposite]), r)
        #         else:
        #             r = min(min(X[group]), r)
            
        #     print(group, opposite)
        l, r = get_y_range(L, V[:, i:], X, independents[i:])

        print(f"range = [{l}, {r}]")

        y = rfunc(l, r)
        X = X + V[:, i] * y
        Y[i] = y

    return Y


def show_df(V: np.ndarray, X: np.ndarray, index: list):

    df = pd.DataFrame(V, columns=index)
    df["X"] = X
    print(df)


def step_y(L: int, V: np.ndarray, X: np.ndarray, index: list, rfunc=uniform):

    show_df(V, X, index)
    Y = np.zeros((len(index), ), float)

    mid = 1 << (L-1)
    Y[0] = get_range(V[:, 0], X[:mid]+X[mid:])
    X = X + V[:, 0] * Y[0]

    show_df(V, X, index) 


    return Y


def float_y(L: int, Y: np.ndarray, rfunc=uniform):
    dependents = []
    independents = []
    for i in range(len(Y)):
        if Y[i] == 0:
            independents.append(i)
        else:
            dependents.append(i)

    V = combination_matrix_inv(L)
    X = V[:, dependents].dot(Y[dependents])

    # vy = float_y_range(L, V[:, independents], X, independents, rfunc)
    vy = step_y(L, V[:, independents], X, independents, rfunc)

    R = np.zeros((len(Y), ), float)
    R[dependents] = Y[dependents]
    R[independents] = vy
    return R


def test_float_y(L: int, Y: np.ndarray, rfunc=uniform):
    R = float_y(L, Y, rfunc)
    X = solve_full_rank(R, L)
    rdf = pd.DataFrame({
        "Y": Y,
        "R": R,
        "X": X
    })
    print(rdf)


def test_y_rand(Y: np.ndarray, L: int, independents: list=None):
    if not independents:
        independents = []
        for i in range(len(Y)):
            if Y[i] == 0:
                independents.append(i)

    R = y_rand(
        L, independents, Y, 
        # lambda l, r: (l+r)/2
        lower
        # upper
    )
    X = solve_full_rank(R, L)
    rdf = pd.DataFrame({
        "Y": Y,
        "R": R,
        "X": X
    })
    print(rdf)

    for i, x in enumerate(X):
        if x < 0:
            raise ValueError(f"X{i} = {x} < 0")


def reduce_half(A: np.ndarray, T: np.ndarray, LV: np.ndarray, L: int, begin: int):
    if L <= 0:
        return
    mid = 1 << (L-1)
    adf = pd.DataFrame(A[begin:begin+(1<<L)], list(range(begin, begin+(1<<L))))
    # adf["T"] = T[begin:begin+(1<<L)]
    # adf["LV"] = LV[begin:begin+(1<<L)]
    # print(adf)
    for i in range(begin, begin+mid):
        if T[i] and (LV[i] == LV[i+mid]):
            A[i, :] -= A[i+mid, :]
            # T[i] -= T[i+mid]
            LV[i] -= mid * T[i+mid]
    

    reduce_half(A, T, LV, L-1, begin)
    reduce_half(A, T, LV, L-1, begin+mid)


def reduce_matrix(index: list, L: int):
    size = 1 << L
    isy = np.zeros((size,), int)

    A = np.identity(size, int)
    # A = combination_matrix(L)
    for i in index:
        isy[i] = 1

    LV = np.zeros((size, ), int)
    reduce_half(A, isy, LV, L, 0)

    return A


def sub_tag(A: np.ndarray):
    m, n = A.shape
    T = np.zeros((m, ), int)
    T[0] = 1
    for i in range(1, m):
        for j in range(1, n):
            if not A[i, j] <= A[0, j]:
                T[i] = j
                break

    return T


def is_sub(P: np.ndarray, S: np.ndarray, r: int):
    while r >= 0:
        if not (S[r] <= P[r]):
            return False
        r -= 1

    return True


def ex_combinations(L: int, independents: list):
    EX = expand_matrix(L, independents)
    EY = solveY(EX, L)
    size = 1 << L

    for esize in reversed(range(len(independents)+1)):
        
        for index in combinations(range(len(independents)), esize):
            max_i = independents[index[-1]]
            msize = size - max_i - 1
            asize = size - msize - esize
            targets = np.zeros((size,), int)


            count = 0
            frank = 0
            for acomb in combinations(range(size-msize), asize):
                i = 0
                for a in acomb:
                    targets[i] = a
                    i += 1
                for a in range(max_i+1, size):
                    targets[i] = a
                    i += 1
                for a in index:
                    targets[i] = a + size
                    i += 1
                
                A = EY[:, targets]
                if np.linalg.matrix_rank(A) == size:
                    frank += 1
                else:
                    
                    print([independents[ix] for ix in index])
                    print(targets)
                    print(A)
                    print(EX[:, targets])
                    break
                count += 1
        
            yield index, f"C({size-msize}, {asize}) = {count}, fullrank = {frank}"

        if esize < len(independents):
            break


def start_points(index: list, size: int):
    l = len(index)
    i = 0
    s = 0
    points = np.zeros((size - l,), int)
    p = 0
    while i < l and s < size:
        if s < index[i]:
            points[p] = s
            p += 1
        else:
            i += 1
        s += 1
    
    while s < size:
        points[p] = s
        s += 1
        
    return points        


def expand_combinations(EV: np.ndarray, columns: list, index: list):

    tags = [False] * len(columns)

    size = len(index)
    for tags in iter_expand_tags(EV, index, tags, len(columns)):
        comb = [0] * size
        c = 0
        for i, t in enumerate(tags):
            if t:
                comb[c] = i
                c += 1

        yield comb     
    

def iter_expand_tags(A: np.ndarray, index: list, tags: list, M: int):

    if len(index) == 1:
        n = index[0]
        for m in reversed(range(M)):
            if A[n, m] != 0 and not tags[m]:
                tags[m] = True
                yield tags
                tags[m] = False

        return
    
    n = index[-1]
    for m in reversed(range(M)):
        if A[n, m] != 0 and not tags[m]:
            tags[m] = True
            yield from iter_expand_tags(A, index[:-1], tags, M)
            tags[m] = False


def levelfill(A: np.ndarray, L: int, level: int, start: int=0):
    if level < L:
        mid = 1 << (L-1)
        levelfill(A, L-1, level, start)
        levelfill(A, L-1, level, start+mid)
        return
    
    mid = 1 << (L-1)
    A[start:start+mid] = A[start:start+mid] + A[start+mid:start+mid+mid]


# def show_range(A: np.ndarray, Y: np.ndarray, L: int, n: int):
    


def test_y_range(Y: np.ndarray, L: int, independents: list=None, rfunc=uniform, results: dict=None):
    dependents = []
    if not independents:
        independents = []


        for i in range(len(Y)):
            if Y[i] == 0:
                independents.append(i)
            else:
                dependents.append(i)
    else:
        j = 0
        i = 0
        while i < len(Y) and j < len(independents):
            if i < independents[j]:
                dependents.append(i)
                i += 1
            else:
                i += 1
                j += 1
        
        while i < len(Y):
            dependents.append(i)
            i += 1

        print(dependents)

    A = combination_matrix(L)
    EX = expand_matrix(L, independents)
    EY = Y.copy()
    for i in range(len(Y)):
        if EY[i] == 0:
            EY[i] = 10000
    
    # df = pd.DataFrame(EX)
    # df["Y"] = EY
    # print(df)
    size = 1 << L
    REX = solveY(EX, L)[:, size:]

    rdf = pd.DataFrame(REX, columns=independents)
    EY = solveY(EY, L)
    rdf["Y"] = EY
    # print(edf)
    
    print(rdf)

    mid = 1 << (L-1)

    # for i in range(6):    
    #     REX[i+1, :] += REX[i, :]
    #     EY[i+1] += EY[i]
    #     REX[i+1 + mid, :] += REX[i+mid, :]
    #     EY[i+1+mid] += EY[i+mid]

    # X = np.array(
    #     [0, 0, 0, 0, 0, 0, 9911],
    #     int
    # )

    # xindex = [0, 1, 2, 3, 4, 5]

    # # EY -= REX.dot(X)
    
    # rdf = pd.DataFrame(REX[:, xindex], columns=[independents[i] for i in xindex])
    # rdf["Y"] = EY
    # print(rdf)
    # # mid = 1 << (L-1)

    # rg = pd.DataFrame({
    #     "minimum": -EY[:mid],
    #     "maximum": EY[mid:],
    # })
    # rg["space"] = rg["maximum"] - rg["minimum"]
    # print(rg)

    # for i in []:
    #     levelfill(REX, L, L-i)
    #     levelfill(EY, L, L-i)
    # EY = rdf["Y"].values
    # for i, j in [
    #     (1, 0),
    #     (2, 1),
    #     (3, 2),
    #     (4, 3),
    #     (5, 4),
    #     (6, 5)
    # ]:
    #     REX[i] += REX[j]
    #     EY[i] += EY[j]

    # rdf = pd.DataFrame(REX, columns=independents)
    # rdf["Y"] = EY
    # if results:
    #     for n, x in results.items():
    #         rdf["Y"] -= rdf.pop(n) * x

    #     # x = 9
    #     # for i in [0, 8]:
    #     #     if rdf.loc[i, x] == -1:
    #     #         rdf.loc[i, x] += 1
    #     #         rdf.loc[i, "Y"] += 7214
    #     #     elif rdf.loc[i, x] == 1:
    #     #         rdf.loc[i, x] += -1
    #     #         rdf.loc[i, "Y"] += -5217

    # print(rdf)


def main():

    batch = Batch(
        [0, 1, 2, 3],
        [
                # ([0, 1, 2], 0.1),
            ([0, 1], 0.2),
            ([1, 2], 0.3),
            ([2, 3], 0.25),
            # ([0, 3], 0.25),
            ([0], 0.5),
            ([1], 0.5),
            ([2], 0.5),
            ([3], 0.5),
            # add
            # ([0, 2], 0.2),
            # ([0, 1, 2], 0.1),
            # ([0, 3], 0.3),
            # ([1, 3], 0.4),
            # ([0, 1, 3], 0.2),
            # ([0, 2, 3], 0.2),
            # ([0, 1, 2, 3], 0.1),
            # ([1, 2, 3], 0.25),
        ]
    )

    # batch = Batch(
    #     [0, 1, 2],
    #     [
    #         ([0], 0.5),
    #         ([1], 0.5),
    #         ([2], 0.5),
    #         ([0, 1], 0.25),
    #         ([0, 2], 0.25),
    #     ]
    # )

    # test_basic_solution(batch)
    # test_no_full_rank(5, 0.3)
    # test_rank()
    # test_distance(batch.bisupports(), batch.L)
    # Y = batch.bisupports()
    # test_y_range(Y, batch.L)
    # print(solveY(Y, batch.L))
    # test_y_range(np.array([1, 0.5, 0, 0]), 2)
    # test_fill(batch.L, batch.bisupports())
    
    L = 4
    # # Y = random_y(L)
    # # Y = np.array([1.0, 0.64, 0.53, 0.0, 0.54, 0.36, 0.3, 0.0])
    # # Y = np.array([1.0, 0.64, 0.53, 0.53, 0, 0.39, 0.39, 0.0])
    Y = np.array(
        [1.0, 0.4783623722876001, 0.42135025029614215, 0.0, 0.5590830726336207, 0.27860867671873163, 0.2893586315531935, 0.0, 0.5536892042543795, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # [1.0, 0.4783623722876001, 0.42135025029614215, 0.0089, 0.5590830726336207, 0.27860867671873163, 0.2893586315531935, 0.0089, 0.5536892042543795, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    # L = 3
    # Y = np.array([1, 0.6, 0.6, 0.3, 0.6, 0.3, 0, 0])
    Y = (Y*1e4).astype(int)
    print(list(Y))

    test_y_range(
        Y, L,
        rfunc=lower,
    )



if __name__ == "__main__":
    main()