import numpy as np
np.set_printoptions(suppress=True)
import unittest
from kitti_camera import KittiCamera
from kitti_generator import KittiGenerator, extract_multipatch
from reprojection.reprojection import Reprojection
from scipy import misc
import timeit
import matplotlib.image as mpimg
from dataset_preparation.set_generator import SetGenerator
import os

class CameraParserTest(unittest.TestCase):

    sequences_path = "/Volumes/Bahia/kitti-dataset/sequences"
    calibration_path = "/Volumes/Bahia/kitti-dataset/calibration"
    dataset_path = "/Volumes/Bahia/kitti-dataset"

    def test_parse_kitti(self):

        sequence = "00"
        camera = 0

        kittiCams = KittiCamera(self.calibration_path, sequence)
        cam0 = kittiCams.getCamera(0)
        cam1 = kittiCams.getCamera(1)
        cam2 = kittiCams.getCamera(2)
        cam3 = kittiCams.getCamera(3)

    def test_kitti_generator(self):

        kittiGen = KittiGenerator(self.sequences_path)

        print kittiGen.sequence_names
        print kittiGen.sq_len
        print kittiGen.sq_dimensions

        sq_num, subset = kittiGen.generate_set()
        print subset

        patches = kittiGen.generate_patch(sq_num)
        print patches


    def test_kitti_camera_depth_plane(self):

        kittiCams = KittiCamera(self.calibration_path, "00")
        cam_original = kittiCams.getCamera(0)
        cam_virtual = kittiCams.getCamera(1)

        # reprojection object
        r = Reprojection(cam_original, cam_virtual)

        # generate set
        kittiGen = KittiGenerator(self.sequences_path)
        sq_num, subset = kittiGen.generate_set()
        print subset

        start_time_read = timeit.default_timer()
        image_set = [
            misc.imread(subset[0]),
            misc.imread(subset[1]),
            misc.imread(subset[2]),
            misc.imread(subset[3]),
            misc.imread(subset[4])
        ]

        start_time = timeit.default_timer()
        patches = kittiGen.generate_patch(sq_num)

        patch_set = MultiprocessorExtractor(image_set, patches, r).generate_planes()

        elapsed = timeit.default_timer() - start_time
        print "Elapsed time to extract 96 depth planes X 4 cameras: %.2f seconds" % elapsed
        elapsed = timeit.default_timer() - start_time_read
        print "Elapsed time with image read: %.2f seconds" % elapsed

        print len(patch_set)

        #plt.imshow(result)
        #plt.show()


    def test_multipatch_generation(self):
         set_gen = SetGenerator(os.path.join(self.dataset_path, 'sequences'))
        patches = set_gen.generate_patch("00")

        for key in patches:
            print("Patch key: %s" % key)
            patch = patches[key]
            print("TL:%s BR:%s" % (patch[0], patch[1]))
            center = (patch[0][0] + key / 2, patch[0][1] + key / 2)
            print("Center: (%s,%s)" % center)
        # print(patches)

        image = mpimg.imread('/Volumes/Bahia/kitti-dataset/sequences/00/image_2/000000.png')
        extracted = extract_multipatch(image, patches)

        print("Done")


