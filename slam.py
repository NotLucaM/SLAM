import math
import random

import numpy as np


def ransac(measurements, N=25, S=15, X=10, C=30):
    filtered = []
    for measurement in measurements:
        if measurement[1] == 0:
            filtered.append(False)
        else:
            filtered.append(True)

    measurements = measurements[filtered]

    cartesian = np.array([ptc(xi) for xi in measurements])

    lines = []

    ret = None

    f = True

    for _ in range(N):
        size = len(cartesian)
        if size <= 2 * S - 1:
            return lines, ret
        indexes = [*range(S)]
        indexes.remove(S // 2)
        indexes = np.array(indexes)

        start = random.randint(S, size - S)
        np.random.shuffle(indexes)
        translated = indexes + np.ones(S - 1) * start
        sample_data = np.array([cartesian[int(i)] for i in translated])

        lsrl = np.linalg.lstsq(
            np.vstack([sample_data.T[0],
                       np.ones(len(sample_data.T[0]))]).T,
            sample_data.T[1], rcond=None)[0]

        normal = np.array([1, lsrl[0]])
        normal = normal / np.linalg.norm(normal)
        inter = np.array([0, lsrl[1]])
        filtered = cartesian[abs(np.cross(cartesian - inter, normal)) < X]

        if len(filtered) > C:
            new_lsrl = np.linalg.lstsq(
                    np.vstack([filtered.T[0],
                               np.ones(len(filtered.T[0]))]).T,
                    filtered.T[1], rcond=None)[0]

            lines.append(new_lsrl)

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

    return lines, ret


def ptc(coords):
    return np.array([coords[1] * math.cos(coords[0]), coords[1] * math.sin(coords[0])])
