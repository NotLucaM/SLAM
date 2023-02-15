import math
import random

import numpy as np
import pygame

global walls
walls = np.array([
])


def init():
    global walls
    walls_py = []
    walls_py = add_rect(walls_py, [10, 10], [990, 790])
    walls_py = add_rect(walls_py, [500, 500], [600, 600])
    walls_py = add_rect(walls_py, [250, 150], [300, 250])
    walls_py = add_rect(walls_py, [800, 600], [990, 790])

    walls = np.array(walls_py)


def add_rect(walls, p1, p2):
    max_x = max(p1[0], p2[0])
    min_x = min(p1[0], p2[0])
    max_y = max(p1[1], p2[1])
    min_y = min(p1[1], p2[1])

    walls += [np.array([[min_x, min_y], [max_x, min_y]])]
    walls += [np.array([[max_x, min_y], [max_x, max_y]])]
    walls += [np.array([[max_x, max_y], [min_x, max_y]])]
    walls += [np.array([[min_x, max_y], [min_x, min_y]])]

    return walls


# pos should be [[x, y], [theta]]
def get_lidar(pos, lines=360, noise=0.1, max_range=400, off_chance=0.01):
    ret = np.zeros((360, 2))
    r_theta = pos[1][0]
    origin = pos[0]

    global walls
    for t in range(lines):
        for wall in walls:
            th = (t * 360 / lines) * math.pi / 180

            ret[t][0] = th

            r_vec = np.array([math.cos(r_theta + th), math.sin(r_theta + th)])
            v1 = origin - wall[0]
            v2 = wall[1] - wall[0]
            v3 = np.array([-r_vec[1], r_vec[0]])

            dot = np.dot(v2, v3)
            if abs(dot) < 0.000001:
                continue

            t1 = np.cross(v2, v1) / dot
            t2 = np.dot(v1, v3) / dot

            if 0.0 <= t1 <= max_range and (0.0 <= t2 <= 1.0):
                if ret[t][1] != 0:
                    if random.random() < off_chance:
                        ret[t][1] = random.randint(0, max_range)
                        continue
                    ret[t][1] = min(ret[t][1], t1 + np.random.normal(scale=noise))
                else:
                    ret[t][1] = t1 + np.random.normal(scale=noise)

    return ret


def display(screen):
    global walls
    for wall in walls:
        pygame.draw.line(screen, (255, 0, 0), wall[0], wall[1])


def show_measurements(screen, measurements, origin=np.array([[0, 0], [0]])):
    for measurement in measurements:
        th = measurement[0]
        th += origin[1][0]
        point = np.array([measurement[1] * math.cos(th), measurement[1] * math.sin(th)])
        point += origin[0]

        pygame.draw.circle(screen, (0, 255, 0), point, 1)
