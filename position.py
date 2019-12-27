from models import *
from utils import kalman, tail
from time import sleep
import numpy as np
import math 
import utils
import sys
import time
from scipy.spatial.transform import Rotation 


# 'rotation': np.array([[0.8660254,0,0.5000000], [0,1,0], [-0.5000000,0,0.8660254]])

class Position:
    cameras_config = {
        1: {'type': 'dynamic'}
        #2: {'type': 'static', 'translation': np.array([0, 0, -0.2]), 'rotation': np.array([[0.7071068,0,-0.7071068], [0,1,0], [0.7071068,0,0.7071068]]) }
    }

    markers_config = {
        159: {'type': 'static', 'translation': np.array([0, 0, 0])},
        # #159: {'type': 'static', 'translation': np.array([0, 0, 0]), 'rotation': np.array([[0,0,0], [0,0,0], [0,0,0]])},
        # 114: {'type': 'static', 'translation': np.array([0, 0, 0]), 'rotation': np.array([[0,0,0], [0,0,0], [0,0,0]])},
        155: {'type': 'dynamic'},
        24: {'type': 'dynamic'}
    }

    def get__static_marker_dynamic_marker(self, marker_id, reference_marker_id, camera_id):  # OK
        target_position = utils.get_position_from_camera(marker_id, camera_id)
        reference_position = utils.get_position_from_camera(reference_marker_id, camera_id)

        target_position.translation = target_position.translation.dot(reference_position.rotation_matrix)
        reference_position.translation = reference_position.translation.dot(reference_position.rotation_matrix)
        if 'rotation' in self.markers_config[reference_marker_id]:
            target_position.translation = target_position.translation.dot(self.markers_config[reference_marker_id]['rotation'])
            reference_position.translation = reference_position.translation.dot(self.markers_config[reference_marker_id]['rotation'])

        direction_vector = target_position.translation - reference_position.translation
        direction_vector[2] = -direction_vector[2]
        abs_position = self.markers_config[reference_marker_id]['translation'] + direction_vector
        abs_rotation = target_position.rotation_matrix.dot(reference_position.rotation_matrix)

        return [abs_position, abs_rotation]

    def get__static_marker_dynamic_camera(self, marker_id, camera_id):  # OK
        # ----
        target_position = utils.get_position_from_camera(marker_id, camera_id)
        # ----
        reference_position = utils.get_position_from_camera(marker_id, camera_id)
        reference_position.translation = reference_position.translation.dot(-reference_position.rotation_matrix)
        if 'rotation' in self.markers_config[marker_id]:
            reference_position.translation = reference_position.translation.dot(self.markers_config[marker_id]['rotation'])
        reference_position.translation[2] = -reference_position.translation[2]

        abs_position = self.markers_config[marker_id]['translation'] + reference_position.translation
        abs_rotation = target_position.rotation_matrix.dot(reference_position.rotation_matrix)
        return [abs_position, abs_rotation] 

    def get__dynamic_marker_static_camera(self, marker_id, camera_id):  # OK
        marker_position = utils.get_position_from_camera(marker_id, camera_id)
        if 'rotation' in self.cameras_config[camera_id]:
            marker_position.translation[2] = -marker_position.translation[2]
            marker_position.translation = marker_position.translation.dot(self.cameras_config[camera_id]['rotation'])
            marker_position.translation[2] = -marker_position.translation[2]

        abs_position = self.cameras_config[camera_id]['translation'] + marker_position.translation
        return abs_position

    def get__auto(self, target_type, camera_id, target_id):
        # If target marker is static: return known position
        #print('Entered get__auto from get')
        if target_type == 'marker' and self.markers_config[target_id]['type'] == 'static':
            return [self.markers_config[target_id]['translation'], self.markers_config[target_id]['rotation']]

        # Get markers in frame
        markers_in_frame = utils.get_markers_in_frame(camera_id)
        #print(markers_in_frame)

        if target_type == 'marker':
            if target_id not in markers_in_frame:
                return
            markers_in_frame.remove(target_id)
        
        for aux_marker in markers_in_frame:

            if aux_marker not in self.markers_config:
                continue
            if self.markers_config[aux_marker]['type'] == 'static':
                if target_type == 'marker':
                    # If target marker is dynamic and we see a static: get__static_marker_dynamic_marker
                    #print('get__static_marker_dynamic_marker')
                    return self.get__static_marker_dynamic_marker(target_id, aux_marker, camera_id)
                elif target_type == 'camera':
                    # We want to know the position of the camera: get__dynamic_marker_static_camera
                    #print('get__static_marker_dynamic_camera')
                    return self.get__static_marker_dynamic_camera(aux_marker, camera_id)

        # If target marker is dynamic and camera is static: get__dynamic_marker_static_camera
        if self.cameras_config[camera_id]['type'] == 'static':
            #print('get__dynamic_marker_static_camera')
            return self.get__dynamic_marker_static_camera(target_id, camera_id)

    def get(self, target_type, target_id, cameras_ids):
        calculations = []
        rotation = None

        #print('Entered get')

        for cam_id in cameras_ids:
            [translation, rotation] = self.get__auto(target_type, cam_id, target_id)
            if translation is not None:
                calculations.append(translation)

        x_values = [x[0] for x in calculations]
        y_values = [x[1] for x in calculations]
        z_values = [x[2] for x in calculations]
        return [np.array([np.mean(x_values), np.mean(y_values), np.mean(z_values)]), rotation]


    def live(self, cameras_ids=[1]):

        start = time.time()

        markers = set()

        for cam_id in cameras_ids:
            lines = tail(30, cam_id)
            for line in lines:
                markers.add(int(line.partition(':')[0]))

        while True:
            for marker in markers:
                try:
                    start = time.time()
                    [translation, rotation] = self.get('marker', marker, cameras_ids)

                    # print('---TESTING---')
                    # print('Rotation Matrix to Euler Angles function:')
                    # print(rotation)

                    # print(rotm)
                    # print(utils.isRotationMatrix(rotm))
                    # print(utils.rotationMatrixToEulerAngles(rotm))
                    # print('-------------')

                    # Convert rotation matrix to rotation vector
                    rotation = rotation.dot([[1,0,0], [0,0,1], [0,-1,0]])
                    rotation, _ = cv2.Rodrigues(rotation)
                    print('-------------')
                    print('%s: Translation: [x: %s, y: %s, z: %s]  Rotation vector: [%s, %s, %s]' % (str(marker), round(translation[0], 4), abs(round(translation[2], 4)), round(translation[1], 4), round(rotation[0][0], 4), round(rotation[1][0], 4), round(rotation[2][0], 4)))

                    rotation = np.asarray(rotation).reshape(-1)

                    # psi, theta, phi = utils.euler_angles_from_rotation_matrix(rotation)
                    # print(psi, theta, phi)

                    r = Rotation.from_rotvec(rotation)
                    raseuler = r.as_euler('xyz', degrees=True)
                    print('\n')
                    print('Rotation degrees: %s' % raseuler )
                    print('-------------')

                    # print('---TESTING---')
                    # theta = [rotation[0]**2, rotation[1]**2, rotation[2]**2]
                    # print(theta)

                    # theta = sqrt(rotation[0]**2, rotation[1]**2, rotation[2]**2)
                    # print(theta)

                    # v = [rotation[0]/theta, rotation[1]/theta, rotation[2]/theta]
                    # print(v)

                    # round(rotation[0], 4), round(rotation[1], 4), round(rotation[2], 4)
                    #print('-------------')

                except Exception as e:
                    pass
            sleep(1)
            print(' ')

    def calibrate(self, target_id, cameras_ids):
        [trans, rot] = self.get('marker', target_id, cameras_ids)
        data = {
            'type': 'static',
            'translation': 'np.array([%s, %s, %s])' % (trans[0], trans[1], trans[2]),
            'rotation': 'np.array([[%s, %s, %s], [%s, %s, %s], [%s, %s, %s]])' % (rot[0][0], rot[0][1], rot[0][2], rot[1][0], rot[1][1], rot[1][2], rot[2][0], rot[2][1], rot[2][2],)
        }
        print(data)


if __name__ == '__main__':

    live = sys.argv[1] == 'live'
    calibrate = sys.argv[1] == 'calibrate'

    if live:
        cameras_ids = [int(x) for x in sys.argv[2].split(',')]
        p = Position()
        p.live(cameras_ids)
    else:
        target_type = sys.argv[1]
        target_id = int(sys.argv[2])
        try:
            cameras_ids = [int(x) for x in sys.argv[3].split(',')]
        except:
            cameras_ids = [target_id]
        p = Position()
        if calibrate:
            p.calibrate(target_id, cameras_ids)
        else:
            pos = p.get(target_type, target_id, cameras_ids)
            print('%s: [x: %s, y: %s, z: %s]' % (str(target_id), pos[0][0], pos[0][2], pos[0][1]))
