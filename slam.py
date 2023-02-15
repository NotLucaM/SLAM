import math
import random

import numpy as np


def ransac(measurements, N=100, S=10, X=5, C=75):
    measurements = np.array([measurements])
    print(measurements)
    size = len(measurements)
    if size <= 2 * S:
        return [], None
    indexes = [*range(S)]
    indexes.remove(S // 2)
    indexes = np.array(indexes)

    cartesian = np.array([ptc(xi) for xi in measurements])

    lines = []

    ret = None

    f = True

    for i in range(N):
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

        new_lsrl = None

        if len(filtered) > C:
            new_lsrl = np.linalg.lstsq(
                    np.vstack([filtered[0],
                               np.ones(len(filtered[0]))]).T,
                    filtered[1], rcond=None)[0]

            lines.append(new_lsrl)

        if f and new_lsrl is not None:
            f = False
            ret = [
                [cartesian[start]],
                sample_data,
                filtered,
                lsrl
            ]

    return lines, ret


def ptc(coords):
    return np.array([coords[1] * math.cos(coords[0]), coords[1] * math.sin(coords[0])])
