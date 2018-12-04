"""
This file contains implementation of the core model: Denoising AutoEncoder
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from AE import AutoEncoder, simulate_missing_markets
from utils.data import add_noise, loss_reconstruction
from utils.flags import FLAGS


class FlatAutoEncoder(AutoEncoder):
    """Flat autoencoder.
    It has all-to-all connections at each layer

    The user specifies the structure of the neural net
    by specifying number of inputs, the number of hidden
    units for each layer and the number of final outputs.
    All this information is set in the utils/flags.py file.
    """

    def __init__(self, shape, sess, batch_size, variance_coef, data_info):
        """Autoencoder initializer

        Args:
          shape:          list of ints specifying
                          num input, hidden1 units,...hidden_n units, num outputs
          sess:           tensorflow session object to use
          batch_size:     batch size
          varience_coef:  multiplicative factor for the variance of noise wrt the variance of data
          data_info:      key information about the dataset
        """

        AutoEncoder.__init__(self, len(shape) - 2, batch_size, FLAGS.chunk_length, sess, data_info)

        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]

        self.__variables = {}


        with sess.graph.as_default():

            with tf.variable_scope("AE_Variables"):

                ##############        SETUP VARIABLES       #####################################

                for i in range(self.num_hidden_layers + 1):  # go over all layers

                    # create variables for matrices and biases for each layer
                    self._create_variables(i, FLAGS.Weight_decay)

                if FLAGS.reccurent:

                    # Define LSTM cell
                    lstm_sizes = self.__shape[1:]

                    def lstm_cell(size):
                        basic_cell = tf.contrib.rnn.BasicLSTMCell(
                            size, forget_bias=1.0, state_is_tuple=True)
                        # Apply dropout on the hidden layers
                        if size != self.__shape[-1]:
                            hidden_cell = tf.contrib.rnn.DropoutWrapper\
                                (cell=basic_cell, output_keep_prob=FLAGS.dropout)
                            return hidden_cell
                        else:
                            return basic_cell

                    self._RNN_cell = tf.contrib.rnn.MultiRNNCell(
                        [lstm_cell(sz) for sz in lstm_sizes], state_is_tuple=True)

                ##############        DEFINE THE NETWORK     ###################################

                # Declare a mask for simulating missing_values
                self._mask = tf.placeholder(dtype=tf.float32,
                                            shape=[FLAGS.batch_size, FLAGS.chunk_length,
                                                   FLAGS.frame_size *
                                                   FLAGS.amount_of_frames_as_input],
                                            name='Mask_of_mis_markers')
                self._mask_generator = self.binary_random_matrix_generator(FLAGS.missing_rate)

                # Reminder: we use Denoising AE
                # (http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

                ''' 1 - Setup network for TRAINing '''
                # Input noisy data and reconstruct the original one
                self._input_ = add_noise(self._train_batch, variance_coef, data_info._data_sigma)
                self._target_ = self._train_batch

                # Define output and loss for the training data
                self._output = self.construct_graph(self._input_, FLAGS.dropout)
                self._reconstruction_loss = loss_reconstruction(self._output, self._target_,
                                                                self.max_val)
                tf.add_to_collection('losses', self._reconstruction_loss)
                self._loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

                ''' 2 - Setup network for TESTing '''
                self._valid_input_ = self._valid_batch
                self._valid_target_ = self._valid_batch

                # Define output
                self._valid_output = self.construct_graph(self._valid_input_, 1)

                # Define loss
                self._valid_loss = loss_reconstruction(self._valid_output, self._valid_target_,
                                                       self.max_val)

    def construct_graph(self, input_seq_pl, dropout):

        """ Contruct a tensofrlow graph for the network

        Args:
          input_seq_pl:     tf placeholder for ae input data [batch_size, sequence_length, DoF]
          dropout:          how much of the input neurons will be activated, value in range [0,1]
        Returns:
          Tensor of output
        """

        network_input = simulate_missing_markets(input_seq_pl, self._mask, self.default_value)

        if FLAGS.reccurent is False:
            last_output = network_input[:, 0, :]

            numb_layers = self.num_hidden_layers + 1

            # Pass through the network
            for i in range(numb_layers):
                # First - Apply Dropout
                last_output = tf.nn.dropout(last_output, dropout)

                w = self._w(i + 1)
                b = self._b(i + 1)

                last_output = self._activate(last_output, w, b)

            output = tf.reshape(last_output, [self.batch_size, 1,
                                              FLAGS.frame_size * FLAGS.amount_of_frames_as_input])

        else:
            output, last_states = tf.nn.dynamic_rnn(
                cell=self._RNN_cell,
                dtype=tf.float32,
                inputs=network_input)

            # Reuse variables
            # so that we can use the same LSTM both for training and testing
            tf.get_variable_scope().reuse_variables()

        return output

    # Make more comfortable interface to the network weights

    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]

    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]

    @property
    def shape(self):
        return self.__shape

    @staticmethod
    def _activate(x, w, b, transpose_w=False):
        y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
        return y

    def __getitem__(self, item):
        """Get autoencoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
         item: string, variables internal name
        Returns:
         Tensorflow variable
        """
        return self.__variables[item]

    def __setitem__(self, key, value):
        """Store a tensorflow variable

        NOTE: Don't call this explicitly. It should
        be used only internally when setting up
        variables.

        Args:
          key: string, name of variable
          value: tensorflow variable
        """
        self.__variables[key] = value

    def _create_variables(self, i, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if 'wd' is specified.
        If 'wd' is None, weight decay is not added for this Variable.

        This function was taken from the web

        Args:
          i: number of hidden layer
          wd: add L2Loss weight decay multiplied by this float.
        Returns:
          Nothing
        """

        # Initialize Train weights
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.multiply(2.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        name_w = self._weights_str.format(i + 1)
        self[name_w] = tf.get_variable(name_w,
                                       initializer=tf.random_uniform(w_shape, -1 * a, a))

        # Add weight to the loss function for weight decay
        if wd is not None and FLAGS.reccurent == False:

            if i == 1:
                print('We apply weight decay')

            weight_decay = tf.multiply(tf.nn.l2_loss(self[name_w]), wd, name='w_'+str(i)+'_loss')
            tf.add_to_collection('losses', weight_decay)

        # Add the histogram summary
        tf.summary.histogram(name_w, self[name_w])

        # Initialize Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        self[name_b] = tf.get_variable(name_b, initializer=tf.zeros(b_shape))

        if i < self.num_hidden_layers:
            # Hidden layer fixed weights
            # which are used after pretraining before fine-tuning
            self[name_w + "_fixed"] = tf.get_variable\
                (name=name_w + "_fixed", initializer=tf.random_uniform(w_shape, -1 * a, a),
                 trainable=False)
            # Hidden layer fixed biases
            self[name_b + "_fixed"] = tf.get_variable\
                (name_b + "_fixed", initializer=tf.zeros(b_shape), trainable=False)

            # Pre-training output training biases
            name_b_out = self._biases_str.format(i + 1) + "_out"
            b_shape = (self.__shape[i],)
            b_init = tf.zeros(b_shape)
            self[name_b_out] = tf.get_variable(name=name_b_out, initializer=b_init,
                                               trainable=True)

    def run_less_layers(self, input_pl, n, is_target=False):
        """Return result of a net after n layers or n-1 layer, if is_target is true
           This function will be used for the layer-wise pretraining of the AE

        Args:
          input_pl:  tensorflow placeholder of AE inputs
          n:         int specifying pretrain step
          is_target: bool specifying if required tensor
                      should be the target tensor
                     meaning if we should run n layers or n-1 (if is_target)
        Returns:
          Tensor giving pretraining net result or pretraining target
        """
        assert n > 0
        assert n <= self.num_hidden_layers

        last_output = input_pl[:, 0, :]  # reduce dimensionality

        for i in range(n - 1):
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")

            last_output = self._activate(last_output, w, b)

        if is_target:
            return last_output

        last_output = self._activate(last_output, self._w(n), self._b(n))

        out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                             transpose_w=True)

        return out
