import tensorflow as tf
import os
import socket
from datetime import datetime
import traceback
from tf_deep_stereo.shared import conv_relu
import matplotlib.image as mpimg
import numpy as np

def inference_graph():

    with tf.name_scope("inputs"):
        input1 = tf.placeholder(tf.float32, name='inputpicture0', shape=(1, 376, 1241, 3))
        input2 = tf.placeholder(tf.float32, name='inputpicture1', shape=(1, 376, 1241, 3))
        input3 = tf.placeholder(tf.float32, name='inputpicture2', shape=(1, 376, 1241, 3))

    with tf.variable_scope("c1"):
        crelu1 = conv_relu(input1, 1, 3, in_depth=3)
    with tf.variable_scope("c2"):
        crelu2 = conv_relu(crelu1, 1, 3, in_depth=3)

    return input1, crelu2


def lossF(net_out):

    batch_losses = tf.reduce_sum(net_out)
    return batch_losses


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)

    # Create the adagrad optimizer with the given learning rate.
    optimizer = tf.train.AdagradOptimizer(learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op




def run_training():
    """Train MNIST for a number of steps."""
    sess = None
    try:
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

            # Build a Graph that computes predictions from the inference model.
            input, net_out = inference_graph()

            print("Graph built! continuing...")
            # Add to the Graph the Ops for loss calculation.
            loss = lossF(net_out)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = training(loss, 0.001)


            print("Merging summaries continuing...")
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            print("Initialize variables...")
            # Add the variable initializer Op.
            init = tf.initialize_all_variables()

            print("Starting session...")
            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            print("Creating SummaryWritter...")
            summary_name = datetime.now().strftime("%Y_%B_%d_%H_%M_%S")
            summary_name = "%s-%s" % (summary_name, socket.gethostname())
            summary_dir = os.path.join("/Users/boyander/test-tf", summary_name)


            # Run the Op to initialize the variables.
            sess.run(init)

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

            print("Started SummaryWriter -> %s" % summary_dir)
            # And then after everything is built:

            feed_dict = {
                input: np.expand_dims(mpimg.imread('/Volumes/Bahia/kitti-dataset/sequences/00/image_2/000000.png'), axis=0)
            }

            sess.run(train_op, feed_dict=feed_dict)

            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str,0)
            summary_writer.flush()
            # read validation batch

    except Exception as e:
        print("Exception on TRAIN: %s" % e)
        traceback.print_exc()
        if sess:
            sess.close()


def main():
    run_training()

# Run main
if __name__ == "__main__":
    main()
