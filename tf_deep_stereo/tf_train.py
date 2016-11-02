import os
import time
import cv2
import socket
import traceback
from datetime import datetime
from optparse import OptionParser
import tensorflow as tf

from dataset_preparation.kitti_generator import KittiGenerator
from dataset_preparation.queued_processor import GeneratorQueued, KittiParams

from select_tower import select_tower
from color_tower import color_tower
from shared import InputOrganizer

import subprocess

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

def dotpMerge(color_tensors, select_tensor, num_planes=96):
    """
    Reimplement Dotp merge in a more pythonic way
    :param inputs:
    :return:
    """
    with tf.name_scope('dotPMerge'):
        result = []
        selections = tf.split(3, num_planes, select_tensor)
        for i in xrange(num_planes):
            color_tensor = color_tensors[i]
            a = selections[i]
            select_rgb = tf.concat(3, [a, a, a])
            r = tf.mul(color_tensor, select_rgb)
            result.append(r)
        return tf.add_n(result)


def inference(input, num_planes=96, batch_size=1):
    print("Tensorflow DeepEstimation Network: (planes:%s)" % num_planes)
    color = color_tower(input, num_planes=num_planes)
    select = select_tower(input, num_planes=num_planes)
    net_out = dotpMerge(color, select, num_planes=num_planes)


    target = input.get_target_placeholder()
    tf.image_summary('Target',  target, max_images=batch_size)
    tf.image_summary('netout_as_is', net_out, max_images=batch_size)

    # summaries for net output of first image
    image_out = net_out[0, :, :, :]
    net_min = tf.reduce_min(image_out, name="out_minimum")
    net_max = tf.reduce_max(image_out, name="out_maximum")
    net_mean = tf.reduce_mean(image_out, name="out_mean")
    tf.scalar_summary(net_min.op.name, net_min)
    tf.scalar_summary(net_max.op.name, net_max)
    tf.scalar_summary(net_mean.op.name, net_mean)
    tf.histogram_summary('out_histogram', image_out)

    return net_out


def lossF(net_out, target):

    substraction = tf.sub(target, net_out)
    absolute = tf.abs(substraction)
    batch_losses = tf.reduce_sum(absolute, [1, 2, 3])

    return tf.reduce_mean(batch_losses, name="Loss")


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


def do_eval(sess, validation_batch, net_out, target):

    substraction = tf.sub(target, net_out)
    absolute = tf.abs(substraction)
    validation_l1 = tf.reduce_sum(absolute, [1, 2, 3])
    validation_score = tf.reduce_mean(validation_l1, name="validation_loss")
    val_summary = tf.scalar_summary(validation_score.op.name, validation_score)

    val_value = sess.run(validation_score, feed_dict=validation_batch)

    print('Validation Score: %s' % val_value)

    return val_summary



def run_training(FLAGS):
    """Train MNIST for a number of steps."""
    sess = None
    try:
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the images from all camera
            input_organizer = InputOrganizer(batch_size=FLAGS.batch_size,
                                             meanzero=FLAGS.mean_zero,
                                             num_planes=FLAGS.num_planes)

            # Build a Graph that computes predictions from the inference model.
            net_out = inference(input_organizer,
                                num_planes=FLAGS.num_planes,
                                batch_size=FLAGS.batch_size)

            print("Graph built! continuing...")
            # Add to the Graph the Ops for loss calculation.
            loss = lossF(net_out, input_organizer.get_target_placeholder())

            # Add to the Graph the Ops that calculate and apply gradients.
            print("Learning rate is: %s" % FLAGS.learning_rate)
            train_op = training(loss, FLAGS.learning_rate)

            # Add the Op to compare the logits to the labels during evaluation.
            #eval_correct = evaluation(net_out, input_organizer.get_target_placeholder())

            print("Merging summaries continuing...")
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            print("Initialize variables...")
            # Add the variable initializer Op.
            init = tf.initialize_all_variables()

            print("Create a saver for writing training checkpoints...")
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            print("Starting session...")
            # Create a session for running Ops on the Graph.
            #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            print("Creating SummaryWritter...")
            git_hash = get_git_revision_short_hash()
            summary_name = datetime.now().strftime("%Y_%B_%d_%H_%M_%S")
            summary_name = "%s-%s-%s" % (summary_name, socket.gethostname(), git_hash)
            summary_dir = os.path.join(FLAGS.traindir, summary_name)
            #os.mkdir(summary_dir)

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

            print("Started SummaryWriter -> %s" % summary_dir)
            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)


            # read validation batch


            print("Starting multiprocessing queue generator...")
            # IMPORTANT: Define generator to be used
            # Parameters for kitti dataset
            kitti_params = KittiParams(
                FLAGS.kitti_path,
                FLAGS.depth_base,
                FLAGS.depth_step,
                FLAGS.patches_per_set
            )
            generator = GeneratorQueued(
                kitti_params,
                input_organizer,
                batch_size=FLAGS.batch_size,
                extraction_workers=FLAGS.extraction_workers,
                aggregation_workers=FLAGS.aggregation_workers
            )

            print("Reading validation batch...")
            validation_gene = KittiGenerator(FLAGS.kitti_path,
                                              FLAGS.depth_base,
                                              FLAGS.depth_step)
            validation_batch = input_organizer.get_feed_dict([validation_gene.validation_batch(num_set_same_img=FLAGS.batch_size)])
            del validation_gene
            print("Done reading validation Batch!!")

            print("Done! Start training loop, validate and save every (%s steps)..." % FLAGS.validate_step)
            # Start the training loop.
            max_steps = FLAGS.max_steps
            for step in xrange(max_steps):

                # Get images to process in a batch
                start_time1 = time.time()
                feed_dict = generator.get_batch()
                duration_images = time.time() - start_time1

                start_time2 = time.time()
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                duration_net = time.time() - start_time2
                print('=== Step %d ===' % step)
                # Write the summaries and print an overview fairly often.
                if step % FLAGS.print_step == 0:
                    # Print status to stdout.
                    print('=== Step %d: loss = %.2f -> images:(%.3f sec), net:(%.3f sec) ===' % (step, loss_value, duration_images, duration_net))
                    # Update the events file.
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save a checkpoint every 100 iterations
                # and evaluate the model periodically.
                if (step + 1) % FLAGS.validate_step == 0 or (step + 1) == max_steps:
                    print("(Step: %s) Checkpoint, saving model." % step)
                    checkpoint_file = os.path.join(summary_dir, 'checkpoint')
                    saver.save(sess, checkpoint_file, global_step=step)
                    eval_summary = do_eval(sess, validation_batch, net_out, input_organizer.get_target_placeholder())
                    #summary_writer.add_summary(eval_summary, step)
                    #summary_writer.flush()
    except Exception as e:
        print("Exception on TRAIN: %s" % e)
        traceback.print_exc()
        if sess:
            sess.close()


def main():

    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")
    parser.add_option("-l", "--learning-rate",
                      default=0.0001, type="float",
                      help="Learning rate")
    parser.add_option("-s", "--max-steps",
                      default=100000, type="int",
                      help="Max steps to do")
    parser.add_option("-m", "--batch-size",
                      default=5, type="int",
                      help="Batch size")
    parser.add_option("-r", "--patches-per-set",
                      default=1, type="int",
                      help="Same image set extract N patches")
    parser.add_option("-p", "--print-step",
                      default=10, type="int",
                      help="Print every N steps")
    parser.add_option("-i", "--validate-step",
                      default=1000, type="int",
                      help="Validate model every N steps")
    parser.add_option("-n", "--num-processes",
                      default=5, type="int",
                      help="Num processes to use when extracting images")
    parser.add_option("-k", "--kitti-path",
                      default='/imatge/mpomar/work/kitti/kitti-dataset',
                      help="Kitti sequences path")
    parser.add_option("-t", "--traindir",
                      default='/imatge/mpomar/work/tf_train',
                      help="Train directory")
    parser.add_option("-q", "--depth-base",
                      default=25.0, type="float",
                      help="Base depth to start plane sweep")
    parser.add_option("-u", "--depth-step",
                      default=0.25, type="float",
                      help="Extract a plane each base_depth + (depth_step * num_planes) ")
    parser.add_option("-e", "--extraction-workers",
                      default=5, type="int",
                      help="Number of workers to extract PSV")
    parser.add_option("-a", "--aggregation-workers",
                      default=1, type="int",
                      help="Number of workers to aggregate PSV batches")
    parser.add_option("-j", "--num-planes",
                      default=30, type="int",
                      help="Number of planes to use")
    parser.add_option("-g", "--mean-zero", action="store_false",
                      default=True,
                      help="Use mean zero on input images")

    (options, args) = parser.parse_args()

    # Code version from GIT
    print("GIT Hash: %s" % get_git_revision_short_hash())

    # Check OpenCV version >= 3.1.0
    print("OpenCV version -> %s" % cv2.__version__)
    opencv_version = cv2.__version__.split('.')
    assert int(opencv_version[0]) >= 3

    print("Tensorflow Version -> %s" % tf.__version__)
    assert tf.__version__ == '0.9.0'

    run_training(options)

# Run main
if __name__ == "__main__":
    main()
