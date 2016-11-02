import numpy as np
np.set_printoptions(suppress=True)
import unittest
from reprojection import Reprojection
from camera import Camera
from PIL import Image

class ReprojectionTest(unittest.TestCase):

    def test_reproject_back(self):

        srcCam = Camera(
            intrinsics=np.array([
                [2302.852541609168, 0.0, 960.0],
                [0.0, 2302.852541609168, 540.0],
                [0.0, 0.0, 1.0]
            ]),
            rotation=np.eye(3),
            translation=np.array([-80, 0, 0]),
            name="cam0"
        )

        virtualCam = Camera(
            intrinsics=np.array([
                [2302.852541609168, 0.0, 960.0],
                [0.0, 2302.852541609168, 540.0],
                [0.0, 0.0, 1.0]
            ]),
            rotation=np.eye(3),
            translation=np.array([-60, 0, 0]),
            name="cam1"
        )

        imgsrc = Image.open("../Dancer/Dancer_c1_frame.jpg")
        image = np.array(imgsrc.getdata(), np.uint8).reshape(imgsrc.size[1], imgsrc.size[0], 3)
        r = Reprojection(srcCam, virtualCam, image)
        result = r.reproject(2000)

        # Save result array
        imgdst = Image.fromarray(result, mode='RGBA')
        imgdst.save("../Dancer/Dancer_cam0Fromcam1.jpg")


