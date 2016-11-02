from load_camera import camera_0
import numpy as np
np.set_printoptions(suppress=True)
import unittest

class TestStringMethods(unittest.TestCase):
    @unittest.skip("demonstrating skipping")
    def test_intrinsic_opengl(self):
        opengl_in = camera_0.get_intrinsic_opengl()
        self.assertEqual(opengl_in.shape, (4, 4))
        print "Intrinsic Matrix converted to opengl"
        print opengl_in

    @unittest.skip("demonstrating skipping")
    def test_extrinsic_opengl(self):
        opengl_ex = camera_0.get_extrinsic_opengl()
        self.assertEqual(opengl_ex.shape, (4, 4))
        print "Extrinsic matrix converted to opengl"
        print opengl_ex

    @unittest.skip("demonstrating skipping")
    def test_opengl_projection(self):
        opengl_proj = camera_0.projection_opengl()
        self.assertEqual(opengl_proj.shape, (4, 4))
        print "Opengl projection matrix from calibrated cameras"
        print opengl_proj

    def test_principal_point(self):
        point = camera_0.get_principal_point()
        print point

    def test_planeCalculaton(self):
        camera_0.camera_plane_at(50.0)


