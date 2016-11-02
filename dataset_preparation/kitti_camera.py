from reprojection.camera import Camera
import os
import numpy as np

class KittiCamera:

    def __init__(self, calibration_path, poses_path):
        self.calibration_path = calibration_path
        self.poses_path = poses_path

        self.rotations = {}
        self.translations = {}
        self.intrinsics = {}
        self.poses = {}

        for s in xrange(11):

            # Cache calibration data
            sq_name = "%02d" % s

            if s != 3:
                self.rotations[sq_name], self.translations[sq_name], self.intrinsics[sq_name] \
                    = KittiCamera.read_calibration_file(self.calibration_path, sq_name)

            self.poses[sq_name] = KittiCamera.read_poses_file(poses_path, sq_name)


    @staticmethod
    def read_calibration_file(calibration_path, sequence_name):
        """
        Read calibration file from Kitti dataset
        :param self:
        :param calibration_path:
        :param sequence_name:
        :return:
        """
        # Read calibration file
        filename = os.path.join(calibration_path, sequence_name, "calib_cam_to_cam.txt")
        variables = dict()
        with open(filename, 'r') as f:
            for line in f:
                key, value = KittiCamera.read_line(line)
                variables[key] = value

        rotations = {}
        translations = {}
        intrinsics = {}

        for camera in ["00", "01", "02", "03"]:
            r = [float(x) for x in variables['R_' + camera]]
            t = [float(x) for x in variables['T_' + camera]]
            k = [float(x) for x in variables['K_' + camera]]
            rotations[camera] = np.reshape(np.array(r), (3, 3))
            translations[camera] = np.array(t)
            intrinsics[camera] = np.reshape(np.array(k), (3, 3))

        return rotations, translations, intrinsics

    @staticmethod
    def read_line(line):
        key, val = line.split(': ', 1)
        val = val.rstrip().split(' ')
        return key, val

    @staticmethod
    def read_poses_file(poses_path, sq_name):
        pose_filename = "%s.txt" % sq_name
        with open(os.path.join(poses_path, pose_filename), 'r') as f:
            matrixes = []
            for l in f.readlines():
                nums = np.array([float(s) for s in l.split(' ')]).reshape(3, 4)
                matrixes.append(nums)
        return matrixes

    def getCamera(self, camera):

        if camera > 3 or camera < 0:
            raise NameError("Camera should be an int from 0 to 3")

        return Camera(self.rotations[camera],
                      self.translations[camera],
                      self.intrinsics[camera])

    def getNamedCamera(self, cam_name, sequence, image_number):

        if not isinstance(sequence, int):
            print("Sequence variable should be int from 0 to 10")
            raise ValueError("Sequence variable should be int from 0 to 10")
        if sequence < 0 or sequence > 10:
            print("Sequence variable should be int from 0 to 10")
            raise ValueError("Sequence variable should be int from 0 to 10")

        if cam_name not in ["00", "01", "02", "03"]:
            print("Camera should be image_2 or image_3")
            raise ValueError("Camera should be image_2 or image_3")

        sq_name = "%02d" % sequence

        pose = self.poses[sq_name][image_number]
        R = pose[:3, :3]
        T = pose[:, 3]

        return Camera(R,
                      T,
                      self.intrinsics[sq_name][cam_name])


