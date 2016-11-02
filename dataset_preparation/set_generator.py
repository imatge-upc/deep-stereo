from PIL import Image
import glob
import os
import random
import numpy as np

class KittiSet():

    def __init__(self, sq_num, cam_name, subset, patches):
        self.sq_number = sq_num
        self.sq_name = '%02d' % sq_num
        self.camera_name = cam_name
        self.subset = subset
        self.patches = patches


class SetGenerator(object):

    def __init__(self, sequences_path):

        self.sequences_path = sequences_path

        # get sequences info
        self.sequence_names = [str(i).zfill(2) for i in range(11)]
        self.num_sq = len(self.sequence_names)
        self.sequences_info = {}
        for sq_name in self.sequence_names:
            self.sequences_info[sq_name] = self.get_sequence_info(sq_name)

    def get_sequence_path(self, sequence):
        return os.path.join(self.sequences_path, sequence)

    def compose_filename(self, sq_name, cam_name, image_num):
        if cam_name == "02":
            cam_name = "image_2"
        elif cam_name == "03":
            cam_name = "image_3"
        else:
            raise ValueError("Only color images from kitti dataset are allowed!"
                             )
        fname = "%s/%s/%06d.png" %(sq_name, cam_name, image_num)
        return os.path.join(self.sequences_path, fname)

    def get_sequence_info(self, sequence_name):
        sq_path = self.get_sequence_path(sequence_name)
        num_images_sq_cam2 = len(glob.glob(os.path.join(sq_path, 'image_2') + '/*.png'))
        num_images_sq_cam3 = len(glob.glob(os.path.join(sq_path, 'image_3') + '/*.png'))

        if num_images_sq_cam2 != num_images_sq_cam3 or num_images_sq_cam2 == 0:
            raise Exception("Error, cameras from sequence %s do not have same length (%s,%s) or are zero."
                            % (sequence_name, num_images_sq_cam2, num_images_sq_cam3))

        first_image = self.compose_filename(sequence_name, '02', 0)
        with Image.open(first_image) as im:
            w, h = im.size

        return {
            'len': num_images_sq_cam2,
            'size': {
                'w': w,
                'h': h
            }
        }

    def get_set_filenames(self, kitti_set):
        image_names = [
            self.compose_filename(kitti_set.sq_name, kitti_set.camera_name, kitti_set.subset[0]),
            self.compose_filename(kitti_set.sq_name, kitti_set.camera_name, kitti_set.subset[1]),
            self.compose_filename(kitti_set.sq_name, kitti_set.camera_name, kitti_set.subset[2]),
            self.compose_filename(kitti_set.sq_name, kitti_set.camera_name, kitti_set.subset[3]),
            self.compose_filename(kitti_set.sq_name, kitti_set.camera_name, kitti_set.subset[4])
        ]
        return image_names

    def generate_patch(self, sq_num, random_center=True):

        patch_original_sizes = {
            't': 8,
            'ps1': 20,
            'ps2': 40,
            'ps3': 60,
            'ps4': 80
        }

        # caclulate margin to do not get patches outside image
        max_size = max([patch_original_sizes[p] for p in patch_original_sizes])
        margin = max_size/2

        dimensions = (
            self.sequences_info[sq_num]['size']['w'],
            self.sequences_info[sq_num]['size']['h']
        )

        # Coordinates are (x, y)
        if random_center:
            center = (
                random.randint(margin, dimensions[0] - margin),
                random.randint(margin, dimensions[1] - margin)
            )
        else:
            center = (
                438.0,
                205.0
            )

        #print("Center is:")
        #print(center)
        points = dict()
        for p_key in patch_original_sizes:
            size_margin = patch_original_sizes[p_key] / 2

            # Coordinates are (x, y) (topL, bottomR)
            points[p_key] = [
                [int(center[0]-size_margin), int(center[1]-size_margin)],
                [int(center[0]+size_margin), int(center[1]+size_margin)],
            ]
        #print points
        return points

    def random_set(self, use_n_cameras=5, num_patches=1):
        # get a random sequence but avoid sequence 03
        sq_num = 3
        while sq_num == 3:
            sq_num = random.randint(0, self.num_sq - 1)
        sq_name = "%02d" % sq_num

        # from this sequence get a random start seed
        sq_len = self.sequences_info[sq_name]['len']
        sq_image_start = random.randint(0, sq_len - use_n_cameras - 1)
        sq_camera_name = random.choice(['02', '03'])

        # get images to get patches from
        subset = range(sq_image_start, (sq_image_start + use_n_cameras) % sq_len)

        patches = {}
        for n in xrange(num_patches):
            patches[n] = self.generate_patch(sq_name)

        # print info about extracted patches
        print("Random: (SQ -> %s) - %s - (reuse_img_patches: %s)"
              % (sq_num, subset, num_patches))

        return KittiSet(sq_num, sq_camera_name, subset, patches)

    def validation_set(self, use_n_cameras=5, num_patches=1):
        sq_num = 4
        sq_camera_name = '02'
        subset = range(0, use_n_cameras)
        sq_name = "%02d" % sq_num
        patches = {}
        for n in xrange(num_patches):
            patches[n] = self.generate_patch(sq_name, random_center=False)

        # print info about extracted patches
        print("Validation: (SQ -> %s) - %s - (reuse_img_patches: %s)"
              % (sq_num, subset, num_patches))
        return KittiSet(sq_num, sq_camera_name, subset, patches)


