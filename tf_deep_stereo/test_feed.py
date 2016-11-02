import os
import unittest
import time
import numpy as np
import tensorflow as tf

from dataset_preparation.kitti_generator import KittiGenerator, save_batch_images, reprojected_images_plot
from dataset_preparation.queued_processor import GeneratorQueued, KittiParams

from tf_deep_stereo.tf_train import dotpMerge
from tf_deep_stereo.shared import InputOrganizer


class GeneratorTest(unittest.TestCase):

    depth_base = 25.0
    depth_step = 0.25
    batch_size = 2
    dataset_path = "/Volumes/Bahia/kitti-dataset"

    def test_generate_batch(self):

        generator = KittiGenerator(
                            self.dataset_path,
                            self.depth_base,
                            self.depth_step)

        start_time = time.time()
        num_set_same_img = 1
        batch = generator.validation_batch(multipatch=False, num_set_same_img=num_set_same_img)
        duration = time.time() - start_time
        print("PSV Time for batch with %s patches per image: (%.3f sec)"
              % (num_set_same_img, duration))

        print("Batch shape is -> %s" % str(batch['target'].shape))

        # Calculate batch size in bytes to have an approximation of GPU usage
        size = sum([batch['planes'][key].nbytes for key in batch['planes']])
        print("Batch size: %s bytes" % size)

        # Save batch to files
        save_to = "/Users/boyander/MASTER_CVC/DepthEstimation/DepthEstimation/batch_test"
        print("saving batch: %s" % save_to)
        reprojected_images_plot(batch)
        #save_batch_images(batch, save_to)
        print("Done saving batch!")

    def test_meanzero(self):

        input_organizer_zero = InputOrganizer(batch_size=self.batch_size, meanzero=True)
        input_organizer_normal = InputOrganizer(batch_size=self.batch_size, meanzero=False)

        generator = KittiGenerator(
                            self.dataset_path,
                            self.depth_base,
                            self.depth_step)
        batch = generator.next_batch(multipatch=True)

        b_zero = input_organizer_zero.get_feed_dict([batch])
        b_normal = input_organizer_normal.get_feed_dict([batch])

        mean_zero = np.mean(b_zero[input_organizer_zero.get_target_placeholder().name])
        mean_normal = np.mean(b_normal[input_organizer_normal.get_target_placeholder().name])
        print("Mean on target is -> (%s for organizer meanzero)"
              " (%s for organizer normal)" % (mean_zero, mean_normal))

        sub_zero = np.sum(np.subtract(np.abs(mean_zero), np.abs(mean_zero)))

        print("Zero substraction is %s" % (sub_zero))


    def test_multiworker(self):

        input_organizer = InputOrganizer(batch_size=self.batch_size, meanzero=True)

        # Parameters for kitti dataset
        kitti_params = KittiParams(
            self.dataset_path,
            self.depth_base,
            self.depth_step,
            patches_per_set=2
        )

        generator = GeneratorQueued(
            kitti_params,
            input_organizer
        )

        for i in xrange(100):
            start_time = time.time()
            batch = generator.get_batch()
            duration = time.time() - start_time
            print("%s - Time for batch of %s PSV: (%.3f sec)" % (i, 5, duration))

    def test_dot_p_merge(self):

        a = tf.Variable(
            [[1.0, 2.0],
             [3.0, 4.0]]
        )
        a = tf.expand_dims(tf.expand_dims(a, 0), 3)
        a = tf.concat(3, [a, a, a])

        b = tf.Variable([[0.5, 0.5],
                         [0.5, 0.5]])
        b = tf.expand_dims(tf.expand_dims(b, 0), 3)
        dotp = dotpMerge([a], b, numplanes=1)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        result = sess.run(dotp)

        print(result)





