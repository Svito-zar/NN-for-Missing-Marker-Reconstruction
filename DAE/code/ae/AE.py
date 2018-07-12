from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils.flags import FLAGS


class AutoEncoder(object):
    """ Generic denoising autoencoder (AE).

    In denoising AE noise is injected to the input and the network tries to recover original data
    More detail can be founded in the original paper:
    http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

    It is am empty class
    to be parent for the Flat and Hierarchical AE.
    (Hierarchical AE has been removed from the repository, because it was not used in the paper)
    """
    _weights_str = "weights{0}"
    _biases_str = "biases{0}"

    def __init__(self, num_hidden_layers, batch_size, sequence_length, sess, data_info):
        """Autoencoder initializer

        Args:
          num_hidden_layers:   number of hidden layers
          batch_size:          batch size
          sequence_length:     length of the sequence which will be feeded into LSTM as once
          sess:                tensorflow session object to use
        """

        self.__num_hidden_layers = num_hidden_layers
        self.__batch_size = batch_size
        self.__sequence_length = sequence_length

        self.__variables = {}
        self.__sess = sess

        self.__scaling_factor = 0.1
        self.__default_value = FLAGS.defaul_value

        self.__max_val = data_info._max_val  # maximal value in the dataset

        #################### Add the DATASETS to the GRAPH ###############3

        #### 1 - TRAIN ###
        self._train_data_initializer = tf.placeholder(dtype=tf.float32,
                                                      shape=data_info._train_shape)
        self._train_data = tf.Variable(self._train_data_initializer, trainable=False,
                                       collections=[], name='Train_data')

        train_frames = tf.train.slice_input_producer([self._train_data],
                                                         num_epochs=FLAGS.training_epochs)

        self._train_batch = tf.train.shuffle_batch\
            (train_frames, batch_size=FLAGS.batch_size, capacity=5000,
             min_after_dequeue=1000, name='Train_batch')

        #### 2 - VALIDATE, can be used as TEST ###
        # (When we are optimizing hyper-parameters, this dataset stores as a validation dataset,
        #  when we are testing the system, this dataset stores a test dataset )
        self._valid_data_initializer = tf.placeholder(dtype=tf.float32, shape=data_info._eval_shape)
        self._valid_data = tf.Variable(self._valid_data_initializer, trainable=False,
                                       collections=[], name='Valid_data')
        valid_frames = tf.train.slice_input_producer([self._valid_data],
                                                     num_epochs=FLAGS.training_epochs)
        self._valid_batch = tf.train.shuffle_batch\
            (valid_frames, batch_size=FLAGS.batch_size, capacity=5000,
             min_after_dequeue=1000, name='Valid_batch')

    def construct_graph(self, input_seq_pl, dropout, test=False, just_middle=False):

        """Get the output of the autoencoder

        This is a dummy function, it has to be defined in each specific class, inheriting class AE

        Args:
          input_seq_pl:     tf placeholder for ae input data: [batch_size, sequence_length, DoF]
          dropout:          how much of the input neurons will be activated, value in [0,1]
          just_middle :     indicate if we want to extract only the middle layer of the network
        """

        return input_seq_pl

    def binary_random_matrix_generator(self, prob_of_missing):
        """ Generate a binary matrix with random values: 0s for missign markers and 1s otherwise
            Each joint is either completely missing or present:
            all 3 coordinates are either given or not.

          Args:
            prob_of_missing:  probability to have a missing marker, also called "missing rate"
          Returns:
            mask : binary matrix to be multiplied on input in order to simulate missing markers
        """

        random_size = [FLAGS.batch_size, FLAGS.chunk_length,
                       int(FLAGS.frame_size * FLAGS.amount_of_frames_as_input / 3)]
        tensor_size = [FLAGS.batch_size, FLAGS.chunk_length, FLAGS.frame_size *
                       FLAGS.amount_of_frames_as_input]

        # Make sure that all coordinates of each point are either missing or present
        random_missing_points = tf.random_uniform(random_size)
        stacked_coords = tf.stack([random_missing_points, random_missing_points,
                                   random_missing_points], axis=3)
        # Make every 3 markers being the same
        stacked_coords = tf.transpose(stacked_coords, perm=[0, 1, 3, 2])

        random_missing_coords = tf.reshape\
            (stacked_coords, [tf.shape(stacked_coords)[0], tf.shape(stacked_coords)[1], -1])

        mask = tf.where(random_missing_coords < 1 - prob_of_missing,
                        tf.ones(tensor_size), tf.zeros(tensor_size))

        return mask

    @property
    def num_hidden_layers(self):
        return self.__num_hidden_layers

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def scaling_factor(self):
        return self.__scaling_factor

    @property
    def default_value(self):
        return self.__default_value

    @property
    def sequence_length(self):
        return self.__sequence_length

    @property
    def session(self):
        return self.__sess

    @property
    def max_val(self):
        return self.__max_val

    @staticmethod
    def _activate(x, w, b, transpose_w=False):
        y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
        return y


def simulate_missing_markets(input_position, mask, const):
    """ Simulate missing markers, by multiplying input on the binary matrix 'mask'

      Args:
        input_position: full body position
        mask:           binary matrix of missing values
        const:          constant to put in place of missing markers
      Returns:
        output :        position, where some markers were replaced by a contrant 'const'
    """

    output = tf.multiply(input_position, mask)

    if const == 0:
        return output
    else:
        default_values = tf.multiply(1 - mask, const)
        output = tf.add(output, default_values, 'Simulate_missing_markers_as_' + str(const))
        return output


def use_existing_markers(inputs, result, mask, const):
    """ Use the information we know instead of the output of the network for not-missing markers

       Args:
         inputs:  the data we have
         result: the output of the network
         mask:   the binary matrix of missing markers
       Returns:
         output : the new body position, which takes the input into account
     """

    # Separate the result of the network network

    result_without_markers_we_had = np.multiply(result, 1 - mask)  # new info
    the_marker_we_had = np.multiply(inputs, mask)  # what we knew before

    if const == 0:
        output = the_marker_we_had + result_without_markers_we_had
    else:
        # We need first to subtract constant value from the "input"
        original_input = inputs - tf.multiply(inputs, 1 - mask)
        # Now we are ready to combine them
        output = original_input + result_without_markers_we_had

    return output


# The following code can be used for  some testing

'''
    random_size = [2,2,2] #[FLAGS.batch_size, FLAGS.chunk_length, int(FLAGS.frame_size / 3)]
    tensor_size = [2,2,6] # [FLAGS.batch_size, FLAGS.chunk_length, FLAGS.frame_size]

    prob_of_missing = 0.2

    data, max_val, mean_pose = read_datasets_from_binary()
    data_info = DataInfo(data.train.sigma, data.train._sequences.shape, data.test._sequences.shape, max_val)
    AE = AutoEncoder(3,16,32,tf.get_default_session(), data_info)

    mask = AE.binary_random_matrix_generator(prob_of_missing)

    input = tf.random_uniform(tensor_size)


    with tf.Session(''):
        input_values = input.eval()

        our_mask = mask.eval()
        #print(input_values, '\n\n\n')

        missing_values = simulate_missing_markets(input_values,our_mask,0).eval()

        result = input.eval()

        result2 = use_existing_markers(missing_values, result, our_mask, 0)

        print(input_values,'\n\n\n')
        print(missing_values,'\n\n\n')
        print('our_mask', our_mask)

        print(result, '\n\n\n')
        print(result2.eval())
'''
