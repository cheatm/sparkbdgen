import numpy as np


def uniform(l, r) -> float:
    return l + np.random.rand() * (r-l)


def random_split(supports: np.ndarray, parts: np.ndarray, randfunc=uniform) -> np.ndarray:
    M = supports.shape[0]
    N = parts.shape[0]

    result = np.zeros((M, N), float)

    part_total_supports = bottle_algo(sum(supports), parts, randfunc) / parts
    # print(part_total_supports)
    temp = supports.copy()
    for n in range(N-1):

        childs = bottle_algo(part_total_supports[n], temp/parts[n], randfunc)
        # print(childs)
        # print(sum(childs))

        result[:, n] = childs[:]
        temp -= childs * parts[n]

    result[:, N-1] = temp / parts[N-1]
    return result


def bottle_algo(volume: float, containers: np.ndarray, randfunc=uniform):
    total_volume = sum(containers)

    if total_volume < volume:
        raise ValueError(f"{total_volume} < {volume}")
    
    n = len(containers)
    result = np.zeros((n,), float)

    for i in range(n):
        c = containers[i]
        total_volume -= c
        r = c

        diff = volume - total_volume
        # print(i, volume, total_volume, diff)
        if diff > 0:
            result[i] += diff
            volume = total_volume
            r -= diff
        r = min(r, volume)
        r = randfunc(0, r)
        result[i] += r
        volume -= r

    return result    


def test_sample():
    supports = np.array([0.1, 0.2, 0.3])
    parts = np.array([0.2, 0.3, 0.5])

    result = random_split(supports, parts, min)
    print(result)

    print(result.dot(parts))


def test_bottle():

    volume = 0.1
    containers = np.array([0.5, 0.3, 0.2])

    result = bottle_algo(volume, containers)

    print(result)

    print(sum(result))

    print(result / containers)


def main():
    test_sample()
    # test_bottle()


if __name__ == "__main__":
    main()


