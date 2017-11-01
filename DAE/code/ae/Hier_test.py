from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from utils.data import read_all_the_data, read_c3d_file, remove_fingers, add_noise
# TODO: what do I actually neeD?
from utils.flags import FLAGS
# import class for both architectures of AE
from HierarchicalAE import HierarchicalAE, loss_reconstruction

class DataInfo(object):
  """Information about the datasets

   Will be passed to the FlatAe for creating corresponding variables in the graph
  """

  def __init__(self, data_sigma, train_shape, eval_shape):
    """DataInfo initializer

    Args:
      data_sigma:   variance in the dataset
      train_shape:  dimensionalities of the train dataset
      test_shape:  dimensionalities of the testing dataset
      eval_shape:  dimensionalities of the evaluation dataset
    """
    self._data_sigma = data_sigma
    self._train_shape =  train_shape
    self._eval_shape = eval_shape

with tf.Graph().as_default() as g:

    layers_width = FLAGS.network_width
    layers_depth = FLAGS.num_hidden_layers
    variance_coef = 0.5
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size

    start_time = time.time()

    # Read the flags
    chunk_length = int(64 * 1.0 / FLAGS.amount_of_frames_as_input)
    pretrain = FLAGS.Layer_wise_Pretraining
    variance = FLAGS.variance_of_noise

    # Read the data
    data, max_val, mean_pose = read_all_the_data()

    # Get the information about the dataset
    data_info = DataInfo(data.train.sigma, data.train._sequences.shape, data.test._sequences.shape)

    # Allow tensorflow to change device alocation when needed
    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
    # Adjust configuration so that multiple execusions are possible
    config.gpu_options.allow_growth = True
    # Start a session
    sess = tf.Session(config=config)

    ae = HierarchicalAE(layers_width, layers_depth, variance_coef, data_info, sess)

    print('\nHierarchical AE was created !')

    '''sess.run(tf.local_variables_initializer())  # to initialize input_producer

    with tf.variable_scope("Train") as main_scope:

        ##############        DEFINE  Optimizer and training OPERATOR      ####################################

        # Define optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # could be FLAGS.training_learning_rate

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ae._loss, tvars), 1e12)
        train_op = optimizer.apply_gradients(zip(grads, tvars),
                                             global_step=tf.contrib.framework.get_or_create_global_step())

        # Create a saver
        saver = tf.train.Saver()  # saver = tf.train.Saver(variables_to_save)

        # Prepare for making a summary for TensorBoard
        train_error = tf.placeholder(dtype=tf.float32, shape=(), name='train_error')
        test_error = tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')
        tf.summary.scalar('Train_error', train_error)
        train_summary_op = tf.summary.merge_all()
        test_summary_op = tf.summary.scalar('Validation_error', test_error)

        summary_dir = FLAGS.summary_dir
        summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())

        num_batches = int(data.train._num_chunks / batch_size)
        num_test_batches = int(data.test._num_chunks / batch_size) - 1

        # Initialize the part of the graph with the input data
        sess.run(ae._train_data.initializer,
                 feed_dict={ae._train_data_initializer: data.train._sequences})
        sess.run(ae._valid_data.initializer,
                 feed_dict={ae._valid_data_initializer: data.test._sequences})

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if (FLAGS.Layer_wise_Pretraining):
            layers_amount = len(ae_shape) - 2

            # create an optimizers
            pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # Make an array of the trainers for all the layers
            trainers = [pretrain_optimizer.minimize(loss_reconstruction(ae.run_less_layers(ae._input_, i + 1),
                                                                        ae.run_less_layers(ae._input_, i + 1,
                                                                                           is_target=True)),
                                                    global_step=tf.contrib.framework.get_or_create_global_step(),
                                                    name='Layer_wise_optimizer_' + str(i)) for i in
                        xrange(len(ae_shape) - 2)]

            # Initialize all the variables
            sess.run(tf.global_variables_initializer())

        else:
            # Initialize variables
            sess.run(tf.global_variables_initializer())


        # A few initialization for the early stopping
        delta = 0.08  # error tolerance for early stopping
        best_error = 10000
        num_valid_batches = int(data.test._num_chunks / batch_size)

        try:

            step = 0

            # Train the whole network jointly
            print('\nWe train on ', num_batches, ' batches with ', batch_size, ' training examples in each for',
                  FLAGS.training_epochs, ' epochs...')
            print("")
            print(" ______________ ______________")
            print("|     Epoch    | Normalized MSE|")
            print("|------------  |-------------- |")

            while not coord.should_stop():
                print('Go')

        finally:
            print('Training is finisgh')'''