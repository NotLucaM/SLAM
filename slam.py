import math
import random

import numpy as np


def ransac(measurements, N=25, S=5, X=25, C=15):
    filtered = []
    for measurement in measurements:
        if measurement[1] == 0:
            filtered.append(False)
        else:
            filtered.append(True)

    measurements = measurements[filtered]

    cartesian = np.array([ptc(xi) for xi in measurements])
    lines1, segs1, ret = _ransac(cartesian, N, S, X, C)

    return lines1, segs1, ret


def _ransac(cartesian, N=50, S=7, X=10, C=15):
    lines = []
    line_segs = []

    ret = None

    f = True

    for _ in range(N):
        size = len(cartesian)
        if size <= 2 * S - 1:
            return lines, line_segs, ret
        indexes = [*range(S)]
        indexes.remove(S // 2)
        indexes = np.array(indexes)

        start = random.randint(S, size - S)
        np.random.shuffle(indexes)
        translated = indexes + np.ones(S - 1) * start
        sample_data = np.array([cartesian[int(i)] for i in translated])
        sample_data.put(0, cartesian[start])

        lsrl = np.linalg.lstsq(
            np.vstack([sample_data.T[0],
                       np.ones(len(sample_data.T[0]))]).T,
            sample_data.T[1], rcond=None)[0]

        normal = np.array([1, lsrl[0]])
        normal = normal / np.linalg.norm(normal)
        inter = np.array([0, lsrl[1]])
        filtered = cartesian[abs(np.cross(cartesian - inter, normal)) < X]

        groups = [[]]
        o = None
        for i in filtered:
            if o is None:
                o = i
                continue

            if np.linalg.norm(o - i) >= X:
                groups.append([])

            o = i

            groups[len(groups) - 1].append(i)

        max_l = 0
        index = 0
        for i, g in enumerate(groups):
            if len(g) > max_l:
                max_l = len(g)
                index = i

        if len(groups[index]) > C:
            filtered = np.array(groups[index])
            new_lsrl = np.linalg.lstsq(
                    np.vstack([filtered.T[0],
                               np.ones(len(filtered.T[0]))]).T,
                    filtered.T[1], rcond=None)[0]

            lines.append(new_lsrl)
            line_segs.append(np.array([filtered[0], filtered[len(filtered) - 1]]))

            if f:
                f = False
                ret = [
                    [cartesian[start]],
                    sample_data,
                    filtered,
                    new_lsrl
                ]

            filtered = np.logical_not(np.isin(cartesian, filtered).T[0])
            cartesian = cartesian[filtered]

    return lines, line_segs, ret


def ptc(coords):
    return np.array([coords[1] * math.cos(coords[0]), coords[1] * math.sin(coords[0])])
