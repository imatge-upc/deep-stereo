import tensorflow as tf
from shared import input_definition, conv_relu, input_structure

def select_tower(input, num_planes=96):
    plane_concats = []
    with tf.variable_scope("select_tower"):

        with tf.variable_scope("input") as scope:
            for plane in xrange(num_planes):
                if plane == 1:
                    scope.reuse_variables()

                print("Select tower (plane=%s)" % plane)
                c = input_structure(input, plane)

                with tf.variable_scope("conv4-ac"):
                    conv4 = conv_relu(c, size=1, in_depth=40, depth=50)
                with tf.variable_scope("conv5"):
                    conv5 = conv_relu(conv4, size=1, in_depth=50, depth=32)
                with tf.variable_scope("conv6"):
                    conv6 = conv_relu(conv5, size=3, in_depth=32, depth=4)
                plane_concats.append(conv6)

        with tf.variable_scope("convolution"):
            out = tf.concat(3, plane_concats)
            with tf.variable_scope("conv7"):
                conv7 = conv_relu(out, size=3, in_depth=num_planes*4, depth=480)
            with tf.variable_scope("conv8"):
                conv8 = conv_relu(conv7, size=3, in_depth=480, depth=480)
            with tf.variable_scope("conv9_before_tanh"):
                conv9 = conv_relu(conv8, size=3, in_depth=480, depth=num_planes)

            tanh = tf.tanh(conv9)

            # Some info summaries
            tf.scalar_summary(conv9.op.name, tf.reduce_mean(tf.reduce_sum(conv9, [1, 2, 3])))
            tf.scalar_summary(tanh.op.name, tf.reduce_mean(tf.reduce_sum(tanh, [1, 2, 3])))

            softmaxes = []
            #
            # tanh[:, 0, 0, :]
            # splits = tf.split(3, 96, tanh)
            # split = 0
            # for s in splits:
            #     q = tf.reshape(s, shape=(batch_size, 64))
            #     # softmax bias
            #     b = tf.get_variable("softmax_bias_%s" % split, [64], initializer=tf.constant_initializer(value=0.0))
            #     # do softmax
            #     nnaming = "softmax%s" % split
            #     probs = tf.nn.softmax(q + b, name=nnaming)
            #     #probs = tf.nn.softmax(q, name=nnaming)
            #     m = tf.reshape(probs, shape=[batch_size, 8, 8, 1])
            #     softmaxes.append(m)
            #     if s == 0:
            #         tf.scalar_summary("softmax_sum_topleft", tf.reduce_sum(probs))
            #
            #     split += 1
            # maxes = tf.concat(3, softmaxes, name="select_tower_out")

            col_r = []
            for i in range(8):
                row_r = []
                for j in range(8):
                    ind_name = "%s_%s" % (i, j)
                    #print ind_name
                    bias = tf.get_variable("softmax_bias_%s" % ind_name,
                                        [num_planes], initializer=tf.constant_initializer(value=0.0))
                    subtensor = tanh[:, i, j, :]
                    row_r.append(tf.nn.softmax(subtensor + bias))
                col_r.append(tf.concat(1,[tf.expand_dims(row, 1) for row in row_r]))

            result = tf.concat(1, [tf.expand_dims(col, 1) for col in col_r])

            # view selection tower for first element in batch
            extraction = tf.expand_dims(tf.transpose(result[0, :, :, :], perm=[2, 0, 1]), 3)
            tf.image_summary("sample_0_10",
                             extraction,
                             max_images=num_planes)
    return result
