from collections import deque
from walrus import Database
import numpy as np
import decimal
import json
import math
import os
from models import *


def kalman(values):
    # intial parameters
    sz = (len(values),)  # size of array
    x = -0.37727  # truth value (typo in example at top of p. 13 calls this z)
    z = values

    Q = 1e-5  # process variance

    # allocate space for arrays
    xhat = np.zeros(sz)       # a posteri estimate of x
    P = np.zeros(sz)          # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)     # a priori error estimate
    K = np.zeros(sz)          # gain or blending factor

    R = 0.1**2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, len(values)):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k]+R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat[-1]


def tail(lines, camera_id=None):
    if camera_id:
        filename = 'logAruco_%s.txt' % (str(camera_id),)
    else:
        filename = 'logAruco_different-x.txt'

    with open(filename) as f:
        return deque(f, lines)


def get_coords(marker_id, timestamp=None, camera_id=None):
    target_coords = []

    last_lines = tail(10, camera_id)

    for line in last_lines:
        if not line.startswith('Time:'):
            if line.startswith(str(marker_id) + ':'):
                target_coords.append(line[len(str(marker_id)) + 2:].rstrip())

    return target_coords


def filter_coords(readings):
    x_list = []
    y_list = []
    z_list = []
    rvec1_list = []
    rvec2_list = []
    rvec3_list = []

    for reading in readings:
        x, y, z, rvec1, rvec2, rvec3 = reading.split(' ')
        x_list.append(float(x))
        y_list.append(float(y))
        z_list.append(float(z))
        rvec1_list.append(float(rvec1))
        rvec2_list.append(float(rvec2))
        rvec3_list.append(float(rvec3))

    return ((kalman(x_list), kalman(y_list), kalman(z_list)), (kalman(rvec1_list), kalman(rvec2_list), kalman(rvec3_list)))


def get_position_from_camera(marker_id, camera_id=None, timestamp=None):
    readings = get_coords(marker_id, timestamp, camera_id)
    filtered_coords = filter_coords(readings)
    return Reading(camera_id, marker_id, np.array(filtered_coords[0]), filtered_coords[1])


def get_markers_in_frame(camera_id):
    markers = set()

    for line in tail(10, camera_id):
        if not line.startswith('Time:'):
            markers.add(int(line.partition(':')[0]))

    return markers

def difference_between(coords1, coords2):
    distance = math.sqrt((decimal.Decimal(coords2[0])-decimal.Decimal(coords1[0]))**2 + (decimal.Decimal(coords2[1])-decimal.Decimal(coords1[1]))**2 + (decimal.Decimal(coords2[2])-decimal.Decimal(coords2[2]))**2)

    return distance


def move_robot(motor, steps, ROBOT_POSITION):
    new_position = ROBOT_POSITION + steps

    if new_position < -6 or new_position > 80:
        print('UNSAFE POSITION (%s). DANGER. BREAKING PROGRAM.' % new_position)
        exit(1)

    os.system('sshpass -p admin314 ssh administrador@192.168.1.20 python testv1.py %s %s' % (motor, new_position))
    return new_position


class CamerasPositionFeedConsumer:
    def __init__(self):
        self.db = Database(host='192.168.0.86')
        self.cg = self.db.consumer_group('world_interpreter', ['cameras_position_feed'])
        self.cg.create()
        self.cg.set_id('$')

    def get_position(self, camera_id):
        self.cg.set_id('$')
        while True:
            messages = self.cg.read(block=0)
            for msg in messages:
                reading_cam_id = msg[1][0][1][b'camera_id']
                if int(reading_cam_id) == camera_id:
                    translation = json.loads(msg[1][0][1][b'translation'].decode())
                    rotation = json.loads(msg[1][0][1][b'rotation'].decode())
                    return [translation, rotation]