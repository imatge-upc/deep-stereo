import tensorflow as tf
from shared import input_definition, conv_relu, input_structure


def color_tower(input, num_planes=96):
    plane_concats = []
    with tf.variable_scope("color_tower") as scope:
        for plane in xrange(num_planes):
            if plane == 1:
                scope.reuse_variables()

            print("Color tower (plane=%s)" % plane)
            c = input_structure(input, plane)

            with tf.variable_scope("conv4"):
                conv4 = conv_relu(c, size=3, in_depth=40, depth=96)
            with tf.variable_scope("conv5"):
                conv5 = conv_relu(conv4, size=1, in_depth=96, depth=32)
            with tf.variable_scope("conv6"):
                conv6 = conv_relu(conv5, size=3, in_depth=32, depth=32)
            with tf.variable_scope("conv7"):
                conv7 = conv_relu(conv6, size=5, in_depth=32, depth=3)
            plane_concats.append(conv7)

        # color tower
        planes_view = tf.pack([plane[0, :, :, :] for plane in plane_concats])
        tf.image_summary("color_tower_output",
                         planes_view,
                         max_images=num_planes)

    return plane_concats
