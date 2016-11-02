import os
from skimage.transform import resize
from dataset_preparation.kitti_camera import KittiCamera
from reprojection.reprojection import Reprojection
import numpy as np
from scipy.misc import imsave
import matplotlib.image as mpimg
from dataset_preparation.set_generator import SetGenerator

class KittiGenerator(object):

    def __init__(self, dataset_basepath, base_depth, depth_step):

        # paths for data
        self.sequences_path = os.path.join(dataset_basepath, 'sequences')
        self.calibration_path = os.path.join(dataset_basepath, 'calibration')
        self.pose_path = os.path.join(dataset_basepath, 'poses')
        assert os.path.exists(self.sequences_path) == 1
        assert os.path.exists(self.calibration_path) == 1
        assert os.path.exists(self.pose_path) == 1

        # Sweeping depth params
        # base depth and depth step
        self.base_depth = base_depth
        self.depth_step = depth_step
        d_from = self.base_depth
        d_to = d_from + (self.depth_step * 96.0)
        print("Depth will be swept from (%s meters to %s meters) (Step: %s)" % (d_from, d_to, depth_step))

        # Kitti calibration data cache
        self.kittiCams = KittiCamera(self.calibration_path, self.pose_path)

        # Set generation
        self.set_generator = SetGenerator(self.sequences_path)

    @staticmethod
    def get_reprojector(kitti_cams, kitti_set, subset_src, subset_virtual):

        src_cam = kitti_cams.getNamedCamera(kitti_set.camera_name,
                                                 kitti_set.sq_number,
                                                 kitti_set.subset[subset_src])
        virtual_cam = kitti_cams.getNamedCamera(kitti_set.camera_name,
                                                     kitti_set.sq_number,
                                                     kitti_set.subset[subset_virtual])
        r = Reprojection(
            camSrc=src_cam,
            camVirtual=virtual_cam)
        return r

    def next_batch(self, multipatch=True, num_set_same_img=1):

        kitti_set = self.set_generator.random_set(num_patches=num_set_same_img)
        return self.generate_batch(kitti_set, multipatch=multipatch)

    def validation_batch(self, multipatch=True, num_set_same_img=1):

        kitti_set = self.set_generator.validation_set(num_patches=num_set_same_img)
        return self.generate_batch(kitti_set, multipatch=multipatch)

    def generate_batch(self, kitti_set, multipatch=True):

        camera_reprojectors = [
            KittiGenerator.get_reprojector(self.kittiCams, kitti_set, subset_src=0, subset_virtual=2),
            KittiGenerator.get_reprojector(self.kittiCams, kitti_set, subset_src=1, subset_virtual=2),
            KittiGenerator.get_reprojector(self.kittiCams, kitti_set, subset_src=3, subset_virtual=2),
            KittiGenerator.get_reprojector(self.kittiCams, kitti_set, subset_src=4, subset_virtual=2)
        ]

        # Get a list with image filenames
        image_names = self.set_generator.get_set_filenames(kitti_set)

        # images will be between 0 and 1
        image_set = [
            mpimg.imread(image_names[0]),
            mpimg.imread(image_names[1]),
            mpimg.imread(image_names[2]),  # TARGET CAMERA
            mpimg.imread(image_names[3]),
            mpimg.imread(image_names[4])
        ]
        # generate some patches
        def_patches = kitti_set.patches

        planes = dict()
        for plane in xrange(96):
            depth = self.base_depth + (self.depth_step * plane)
            #print("Extracting plane (%s) set at depth (%s meters)" % (plane, depth))

            image_cameras = {
                'cam0': camera_reprojectors[0].reproject(depth=depth, frame=image_set[0]),
                'cam1': camera_reprojectors[1].reproject(depth=depth, frame=image_set[1]),
                'cam3': camera_reprojectors[2].reproject(depth=depth, frame=image_set[3]),
                'cam4': camera_reprojectors[3].reproject(depth=depth, frame=image_set[4])
            }
            #print("Extraction done")

            if multipatch:
                for cam_key in image_cameras:
                    image = image_cameras[cam_key]
                    item_name = "plane%s_%s" % (plane, cam_key)

                    extracted_patches = []
                    for patch_key in def_patches:
                        patch = def_patches[patch_key]
                        extracted_patches.append(extract_multipatch(image, patch))

                    planes["%s_10" % item_name] = np.concatenate([patch['10'] for patch in extracted_patches], axis=0)
                    planes["%s_12" % item_name] = np.concatenate([patch['12'] for patch in extracted_patches], axis=0)
                    planes["%s_18" % item_name] = np.concatenate([patch['18'] for patch in extracted_patches], axis=0)
                    planes["%s_30" % item_name] = np.concatenate([patch['30'] for patch in extracted_patches], axis=0)
            else:
                for cam_key in image_cameras:
                    item_name = "plane%s_%s" % (plane, cam_key)
                    planes[item_name] = image_cameras[cam_key]

        if multipatch:
            targets = []
            for k in def_patches:
                # target image is the 8x8 patch
                target_patch = def_patches[k]['t']
                targets.append(extract_patch(image_set[2], target_patch))
            im_target = np.concatenate(targets, axis=0)
        else:
            im_target = image_set[2]

        return {
            'planes': planes,
            'target': im_target
        }


def extract_multipatch(image, patches):
    return {
        '10': extract_patch(image, patches['ps1'], 10),
        '12': extract_patch(image, patches['ps2'], 12),
        '18': extract_patch(image, patches['ps3'], 18),
        '30': extract_patch(image, patches['ps4'], 30)
    }


def extract_patch(image, patch, new_size=None):
    cut = image[patch[0][1]:patch[1][1], patch[0][0]:patch[1][0]]
    if new_size:
        cut = resize(cut, (new_size, new_size))
    return np.expand_dims(cut, axis=0)


def save_batch_images(batch, base_path):
    for key in batch['planes']:
        image = np.squeeze(batch['planes'][key])
        imsave(os.path.join(base_path, "%s.png" % key), image)

    target_image = np.squeeze(batch['target'])
    imsave(os.path.join(base_path, "target.png"), target_image)


def reprojected_images_plot(batch):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(1, (4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 4),  # creates 2x2 grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     )

    p = 15
    for i in range(0, 20, 4):
        im_cam0 = np.squeeze(batch['planes']['plane%s_cam0' % p])
        grid[i].imshow(im_cam0)
        im_cam1 = np.squeeze(batch['planes']['plane%s_cam1' % p])
        grid[i+1].imshow(im_cam1)
        im_cam2 = np.squeeze(batch['planes']['plane%s_cam3' % p])
        grid[i+2].imshow(im_cam2)
        im_cam3 = np.squeeze(batch['planes']['plane%s_cam4' % p])
        grid[i+3].imshow(im_cam3)
        p += 3

    #plt.savefig('~/Desktop/foo.png', bbox_inches='tight')
    plt.axis('off')
    plt.show()

