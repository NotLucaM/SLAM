import time

import sys

from pygame.locals import *

from input import *
from slam import ransac

width = 1000
height = 800
screen_color = (50, 150, 100)

vel = 5
omega = 5 * math.pi / 180

omega_drift = 0.01
pos_drift = 0.3

robot_size = np.array([25, 25])
real_position = np.array([[400, 300], [0]])
measured_position = np.array([[400, 300], [0]])


def logic(screen):
    measurements = get_lidar(real_position, noise=2.5)
    show_measurements(screen, measurements, origin=real_position)
    lines, ret = ransac(measurements)

    for line in lines:
        pygame.draw.line(
            screen,
            (100, 255, 0),
            (0, (line[1] + real_position[0][1]) - line[0] * real_position[0][0]),
            (width, line[1] + real_position[0][1] + (width - real_position[0][0]) * line[0]),
            width=3
        )

    # if ret is not None:
    #     for point in ret[1]:
    #         pygame.draw.circle(screen, (200, 200, 200), point + real_position[0], 3)
    #
    #     for point in ret[2]:
    #         pygame.draw.circle(screen, (0, 200, 200), point + real_position[0], 1)
    #
    #     pygame.draw.circle(screen, (255, 255, 0), ret[0][0] + real_position[0], 5)
    #
    #     print(ret[3])
    #     pygame.draw.line(
    #         screen,
    #         (100, 200, 40),
    #         (real_position[0][1], ret[3][1] + real_position[0][1]),
    #         (width, ret[3][1] + real_position[0][1] + width * ret[3][0])
    #     )
    #
    #     time.sleep(1)


def show_robot(screen, position, color=(0, 0, 0)):
    th = position[1][0]
    points = [
        np.array([robot_size[0], robot_size[1]]),
        np.array([robot_size[0], -robot_size[1]]),
        np.array([-robot_size[0], -robot_size[1]]),
        np.array([-robot_size[0], robot_size[1]]),
    ]
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    points = list(map(lambda p: rot @ p, points))
    points = list(map(lambda p: p + position[0] - robot_size / 2, points))
    points = list(map(lambda p: (p[0], p[1]), points))

    pygame.draw.polygon(screen, color, points)


def main():
    screen = pygame.display.set_mode((width, height))
    init()

    while True:
        screen.fill(screen_color)
        display(screen)
        logic(screen)
        show_robot(screen, real_position)
        show_robot(screen, measured_position, color=(0, 0, 50))
        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            real_position[0] += np.array([0, -vel])
            measured_position[0] += np.array([0, -vel]) + np.random.normal(scale=pos_drift, size=(2, ))

        if keys[pygame.K_s]:
            real_position[0] += np.array([0, vel])
            measured_position[0] += np.array([0, vel]) + np.random.normal(scale=pos_drift, size=(2, ))

        if keys[pygame.K_d]:
            real_position[0] += np.array([vel, 0])
            measured_position[0] += np.array([vel, 0]) + np.random.normal(scale=pos_drift, size=(2, ))

        if keys[pygame.K_a]:
            real_position[0] += np.array([-vel, 0])
            measured_position[0] += np.array([-vel, 0]) + np.random.normal(scale=pos_drift, size=(2, ))

        if keys[pygame.K_q]:
            real_position[1] += np.array([-omega])
            measured_position[1] += np.array([-omega]) + np.random.normal(scale=omega_drift, size=1)

        if keys[pygame.K_e]:
            real_position[1] += np.array([omega])
            measured_position[1] += np.array([omega]) + np.random.normal(scale=omega_drift, size=1)

        if keys[pygame.K_ESCAPE]:
            sys.exit()

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)


if __name__ == '__main__':
    main()
