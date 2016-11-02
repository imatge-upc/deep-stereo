import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import math

def conv_relu(input, size, depth, in_depth=None):
    # Create variable named "weights".

    # http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    sqared = math.sqrt(size*size)
    weights = tf.get_variable("weights", (size, size, in_depth, depth),
        initializer=tf.contrib.layers.xavier_initializer())
        #initializer=tf.random_normal_initializer(mean=0.0, stddev=sqared))
        #initializer=tf.constant_initializer(value=0.0))
    bias = tf.get_variable("bias", [depth],
        initializer=tf.constant_initializer(value=0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=True)
    return tf.nn.relu(tf.nn.bias_add(conv, bias))


def input_definition(inputs, size=30, depths=[], filters=[]):
    with tf.variable_scope("conv0_cam0"):
        conv_cam0 = conv_relu(inputs["cam0"], size=5, in_depth=4, depth=64)
    with tf.variable_scope("conv0_cam1"):
        conv_cam1 = conv_relu(inputs["cam1"], size=5, in_depth=4, depth=64)
    with tf.variable_scope("conv0_cam3"):
        conv_cam2 = conv_relu(inputs["cam3"], size=5, in_depth=4, depth=64)
    with tf.variable_scope("conv0_cam4"):
        conv_cam3 = conv_relu(inputs["cam4"], size=5, in_depth=4, depth=64)

    conc_input = tf.concat(3, [conv_cam0, conv_cam1, conv_cam2, conv_cam3])

    with tf.variable_scope("conv1"):
        r1 = conv_relu(conc_input, size=filters[0], in_depth=256, depth=depths[0])
    with tf.variable_scope("conv2"):
        r2 = conv_relu(r1, size=filters[1], in_depth=depths[0], depth=depths[1])

    # only if third element is defined
    if len(depths) == 3:
        with tf.variable_scope("conv3"):
            r3 = conv_relu(r2, size=filters[2], in_depth=depths[1], depth=depths[2])
    else:
        r3 = r2
    out = r3
    return out


def input_structure(input, plane_num):
    with tf.variable_scope("in30"):
        in_30 = input_definition(input.get_for_plane_size(plane_num, 30),
                                 size=30,
                                 depths=[96, 48, 16],
                                 filters=[3, 5, 5])
    with tf.variable_scope("in18"):
        in_18 = input_definition(input.get_for_plane_size(plane_num, 18),
                                 size=18,
                                 depths=[40, 40, 8],
                                 filters=[3, 3, 3])
    with tf.variable_scope("in12"):
        in_12 = input_definition(input.get_for_plane_size(plane_num, 12),
                                 size=12,
                                 depths=[32, 8],
                                 filters=[3, 3])
    with tf.variable_scope("in10"):
        in_10 = input_definition(input.get_for_plane_size(plane_num, 10),
                                 size=10,
                                 depths=[32, 8],
                                 filters=[3, 3])

    c = tf.concat(3, [
        in_30,
        tf.image.resize_images(in_18, 16, 16, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        tf.image.resize_images(in_12, 16, 16, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        tf.image.resize_images(in_10, 16, 16, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ])

    n_out = 1

    mean = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='mean', trainable=False)
    variance = tf.Variable(tf.constant(1.0, shape=[n_out]),
                       name='variance', trainable=False)
    variance_epsilon = tf.Variable(tf.constant(0.0001, shape=[n_out]),
                       name='epsilon', trainable=False)
    offset = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='offset', trainable=False)
    scale = tf.Variable(tf.constant(1.0, shape=[n_out]),
                       name='scale', trainable=False)

    normalized = tf.nn.batch_normalization(c, mean=mean,
                                           variance=variance,
                                           variance_epsilon=variance_epsilon,
                                           offset=offset,
                                           scale=scale)
    return normalized

class InputOrganizer(object):

    def __init__(self, batch_size, num_planes=96, meanzero=False):
        """
        :param meanzero: Use mean zero when retrieving feed dict
        """

        self.meanzero = meanzero

        if self.meanzero:
            print("Input data will be normalized on range [-1, 1]")
        else:
            print("Not using mean 0 range [0.0, 1.0]!")

        self.cameras = ["cam0", "cam1", "cam3", "cam4"]
        self.sizes = [30, 18, 12, 10]
        self.num_planes = num_planes
        self.num_channels = 4

        self.placeholders = {}
        with tf.name_scope("input_placeholders"):

            for plane in xrange(self.num_planes):
                self.placeholders[plane] = {}
                for size in self.sizes:
                    self.placeholders[plane][size] = {}
                    for cam in self.cameras:
                        name = "%s_%s_%s" % (plane, size, cam)
                        self.placeholders[plane][size][cam] = tf.placeholder(tf.float32,
                                                                             name=name,
                                                                             shape=(batch_size, size, size, self.num_channels))

            # Target patch
            self.target = tf.placeholder(tf.float32,
                                            name="target",
                                            shape=(batch_size, 8, 8, 3))

    def get_for_plane_size(self, plane, size):
        input = self.placeholders[plane][size]
        assert len(input) == 4
        return input

    def get_target_placeholder(self):
        return self.target


    def preprocess_batch(self, image_batch):

        #s = image_batch.shape
        #for im_idx in xrange(s[0]):
        #    for ch_idx in xrange(s[3]):
        #        image_batch[im_idx, :, :, ch_idx] = preprocessing.scale(image_batch[im_idx, :, :, ch_idx])

        return (image_batch * 2.0) - 1.0

    def get_feed_dict(self, images_feed):

        feed_dict = {}

        # Target images
        tar = np.concatenate([t['target'] for t in images_feed], axis=0)
        if self.meanzero:
            feed_dict[self.target.name] = self.preprocess_batch(tar)
        else:
            feed_dict[self.target.name] = tar

        # add 4 images with 4 resolutions for each plane (96 planes in total)
        for plane in xrange(self.num_planes):
            for size in self.sizes:
                for cam in self.cameras:
                    item_name = "plane%s_%s_%s" % (plane, cam, size)
                    images = [im['planes'][item_name] for im in images_feed]
                    c_images = np.concatenate(images, axis=0)

                    if self.meanzero:
                        feed_dict[self.placeholders[plane][size][cam].name] = self.preprocess_batch(c_images)
                    else:
                        feed_dict[self.placeholders[plane][size][cam].name] = c_images

        return feed_dict









