import numpy as np
np.set_printoptions(suppress=True)
import unittest
from kitti_camera import KittiCamera

class CameraParserTest(unittest.TestCase):

    sequences_path = "/Volumes/Bahia/kitti-dataset/sequences"
    calibration_path = "/Volumes/Bahia/kitti-dataset/calibration"

    def test_posed_camera(self):

        kittiCams = KittiCamera(self.calibration_path, "00")
        camera0 = kittiCams.getCamera(0)
        camera1 = kittiCams.getCamera(1)
        camera2 = kittiCams.getCamera(2)
        camera3 = kittiCams.getCamera(3)
        print "Camera 0\n", camera0
        print "Camera 1\n", camera1
        print "Camera 2\n", camera2
        print "Camera 3\n", camera3

        print "Done Test posed cameras"





