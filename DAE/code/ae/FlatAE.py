from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils.data import  read_c3d_file, add_noise, loss_reconstruction
from utils.flags import FLAGS
from AE import AutoEncoder, use_existing_markers, simulate_missing_markets

class FlatAutoEncoder(AutoEncoder):
  """Generic deep autoencoder.

  Autoencoder used for full training cycle.
  The user specifies the structure of the neural net
  by specifying number of inputs, the number of hidden
  units for each layer and the number of final outputs.
  """


  def __init__(self, shape, sess, batch_size,  variance_coef, data_info):
    """Autoencoder initializer

    Args:
      shape:          list of ints specifying
                      num input, hidden1 units,...hidden_n units, num logits
      sess:           tensorflow session object to use
      batch_size:     batch size
      varience_coed:  multiplicative factor for the variance of noise wrt the variance of data
      data_sigma:     variance of data
    """

    AutoEncoder.__init__(self, len(shape) - 2, batch_size, FLAGS.chunk_length, sess, data_info)

    self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]

    self.__variables = {}

    if(FLAGS.Weight_decay is not None):
      print('We apply weight decay')

    with sess.graph.as_default():

      with tf.variable_scope("AE_Variables"):

        ##############        SETUP VARIABLES       ###############################################

        # Create a variable to track the global step.
        global_step = tf.Variable(0,name='global_step',trainable=False)

        for i in xrange(self.num_hidden_layers + 1): # go over all layers

          self._create_variables(i, FLAGS.Weight_decay)

        if(FLAGS.reccurent):
            # Define LSTM cell
            lstm_sizes = self.__shape[1:]
            def lstm_cell(size):
                basic_cell = tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=1.0, state_is_tuple=True)
                #Apply dropout on the hidden layers
                if(size!=self.__shape[-1]):
                    hidden_cell = tf.contrib.rnn.DropoutWrapper(cell=basic_cell, output_keep_prob=FLAGS.dropout)
                    return hidden_cell
                else:
                    return basic_cell

            self._RNN_cell = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell(sz) for sz in lstm_sizes], state_is_tuple=True)

        # Get some constants from the flags file
        self._keep_prob = tf.placeholder(tf.float32) #dropout placeholder

        ##############        DEFINE THE NETWORK     ###############################################

        # Declare a mask for simulating missing_values
        self._mask = tf.placeholder(dtype=tf.float32, shape = [FLAGS.batch_size, FLAGS.chunk_length, FLAGS.frame_size * FLAGS.amount_of_frames_as_input], name = 'Mask_of_mis_markers')

        # 1 - Setup network for TRAINing
        self._input_  = add_noise(self._train_batch , variance_coef, data_info._data_sigma)
        self._target_ = self._train_batch #Input noisy data and reconstruct the original one

        # Define output and loss for the training data
        self._output = self.construct_graph(self._input_, FLAGS.dropout) # process batch of sequences. no dropout

        # Normalize the L2 loss
        self._reconstruction_loss =  loss_reconstruction(self._output, self._target_, self.max_val) #/ tf.cast(size, tf.float32) #(batch_size*chunk_length)
        tf.add_to_collection('losses', self._reconstruction_loss) # allow for weight decay
        self._loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # 2 - Setup network for TESTing
        self._valid_input_  = add_noise(self._valid_batch , variance_coef, data_info._data_sigma)
        self._valid_target_ = self._valid_batch
        # Define output
        self._valid_output = self.construct_graph(self._valid_input_, FLAGS.dropout)

        # Define loss
        self._valid_loss =   loss_reconstruction(self._valid_output, self._valid_target_,self.max_val) #/  tf.cast(size, tf.float32) #(batch_size*chunk_length)

  def construct_graph(self, input_seq_pl, dropout, just_middle = False):

          """Get the output of the autoencoder

          Args:
            input_seq_pl:     tf placeholder for ae input data of size [batch_size, sequence_length, DoF]
            dropout:          how much of the input neurons will be activated, value in [0,1]
            just_middle :     will indicate if we want to extract only the middle layer of the network
          Returns:
            Tensor of output
          """

          network_input = input_seq_pl #simulate_missing_markets(input_seq_pl, self._mask, self.default_value)

          if(FLAGS.reccurent == False):
              last_output = network_input[:,0,:]

              numb_layers = self.num_hidden_layers + 1

              # Pass through the network
              for i in xrange(numb_layers):

                  # First - Apply Dropout
                  last_output = tf.nn.dropout(last_output, dropout)

                  w = self._w(i + 1)
                  b = self._b(i + 1)

                      # Debug
                      # print('Matrix: ', w)
                      # print('Input: ', last_output.shape)
                  last_output = self._activate(last_output, w, b)

              output = tf.reshape(last_output, [self.batch_size, 1, FLAGS.frame_size * FLAGS.amount_of_frames_as_input])

          else:
              output, last_states = tf.nn.dynamic_rnn(
                  cell=self._RNN_cell,
                  dtype=tf.float32,
                  sequence_length=[self.sequence_length for i in range(self.batch_size)],
                  inputs=network_input)

          #print('The final result has a shape:', output.shape)

          tf.get_variable_scope().reuse_variables() # so that we can use the same LSTM both for training and testing


          return output

  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n) + suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n) + suffix]

  @property
  def shape(self):
    return self.__shape

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b)) # was sigmoid before
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

    NOTE: Don't call this explicity. It should
    be used only internally when setting up
    variables.

    Args:
      key: string, name of variable
      value: tensorflow variable
    """
    self.__variables[key] = value

  def run_shallow(self, input_pl):
      """Get the output of the autoencoder,if it would consist
         only from the first and the last layer

      Args:
        input_pl:     tf placeholder for ae input data
        dropout:      how much of the input neurons will be activated, value in [0,1]
      Returns:
        Tensor of output
      """
      with tf.name_scope("shallow_run"):

        #First - Apply Dropout
        #last_output = tf.nn.dropout(input_pl, dropout)

        # Apply first layer of the network
        w = self._w(1)
        b = self._b(1)
        last_output = self._activate(input_pl, w, b)

        # then apply last layer of the network
        w = self._w(self.num_hidden_layers + 1)
        b = self._b(self.num_hidden_layers + 1)
        last_output = self._activate(last_output, w, b)

      return last_output

  def process_sequences_shallow(self, input_seq_pl, dropout):
          """Get the output of the autoencoder, reduced to just the first and the last layers

          Args:
            input_seq_pl:     tf placeholder for ae input data of size [batch_size, sequence_length, DoF]
            dropout:          how much of the input neurons will be activated, value in [0,1]
          Returns:
            Tensor of output
          """

          numb_layers = FLAGS.middle_layer

          # Initial state of the LSTM memory.
          self._RNN_state = self._initial_state

          # First - Apply Dropout
          the_whole_sequences = tf.nn.dropout(input_seq_pl, dropout)

          # Take batches for every time step and run them through the network
          # Stack all their outputs
          stacked_outputs = tf.stack( [ self.run_shallow(the_whole_sequences[:,time_st,:]) for time_st in range(self.sequence_length) ])

          # Transpose output from the shape [sequence_length, batch_size, DoF] into [batch_size, sequence_length, DoF]

          output = tf.transpose(stacked_outputs , perm=[1, 0, 2])

          # print('The final result has a shape:', output.shape)

          return output

  def _create_variables(self, i, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      i: number of hidden layer
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Nothing
    """

    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32 #- TODO: integrate it into the code

    # Initialize Train weights
    w_shape = (self.__shape[i], self.__shape[i + 1])
    a = tf.multiply(2.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
    name_w = self._weights_str.format(i + 1)
    self[name_w] = tf.get_variable(name_w,
                              initializer=tf.random_uniform(w_shape, -1 * a, a))

    # Add weight to the loss function for achieving weight decay
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(self[name_w]), wd, name='weight_'+str(i)+'_loss')
      tf.add_to_collection('losses', weight_decay)

    # Add the histogram summary
    tf.summary.histogram(name_w,self[name_w])

    # Initialize Train biases
    name_b = self._biases_str.format(i + 1)
    b_shape = (self.__shape[i + 1],)
    self[name_b] = tf.get_variable(name_b, initializer=tf.zeros(b_shape))

    if i < self.num_hidden_layers:
      # Hidden layer fixed weights (after pretraining before fine tuning)
      self[name_w + "_fixed"] = tf.get_variable(name=name_w + "_fixed",
                                                initializer=tf.random_uniform(w_shape, -1 * a, a),
                                                trainable=False)
      # Hidden layer fixed biases
      self[name_b + "_fixed"] = tf.get_variable(name_b+"_fixed", initializer=tf.zeros(b_shape),
                                                trainable=False)

      # Pretraining output training biases
      name_b_out = self._biases_str.format(i + 1) + "_out"
      b_shape = (self.__shape[i],)
      b_init = tf.zeros(b_shape)
      self[name_b_out] = tf.get_variable(name=name_b_out, initializer=b_init,
                                         trainable=True)


  def run_less_layers(self, input_pl, n, is_target=False):
    """Return net for step n training or target net
    Args:
      input_pl:  tensorflow placeholder of AE inputs
      n:         int specifying pretrain step
      is_target: bool specifying if required tensor
                  should be the target tensor
    Returns:
      Tensor giving pretraining net or pretraining target
    """
    assert n > 0
    assert n <= self.num_hidden_layers

    last_output = input_pl[:,0,:] # reduce dimensionality

    for i in xrange(n - 1):
      w = self._w(i + 1, "_fixed")
      b = self._b(i + 1, "_fixed")

      last_output = self._activate(last_output, w, b)

    if is_target:
      return last_output

    last_output = self._activate(last_output, self._w(n), self._b(n))

    out = self._activate(last_output, self._w(n), self._b(n,"_out"),
                         transpose_w=True) # TODO: maybe try without symmerty

    return out

  '''def write_middle_layer(self, input_seq_file_name, output_seq_file_name, name):
    """ Writing a middle layer into the matlab file

    Args:
      ae:                     ae, middle layer of which we want to save
      input_seq_file_name:    name of the file with the sequence for which we want to get a middle layer
      output_seq_file_name:   name of the file in which we will write a middle layer of the AE
      name:                   name of the  'trial' for these sequence
    Returns:
      nothing
    """
    print('\nExtracting middle layer for a test sequence...')

    with self.__sess.graph.as_default():

      sess = self.__sess

      # get input sequnce
      currSequence = read_c3d_file(input_seq_file_name)

      # define tensors
      input_ = tf.placeholder(dtype=tf.float32,
                                    shape=(None, self.__shape[0]),
                                    name='ae_input_pl')
      # Define on an operator
      middle_op = self.process_sequences(input_ , 1, just_middle = True) # 1 means that we have no dropout

      # for each snippet in a sequence
      # pass through the network untill the middle layer
      middle = sess.run(middle_op, feed_dict={input_: currSequence})

      # save it into a file
      sio.savemat(output_seq_file_name, {'trialId':name, 'spikes':np.transpose(middle)})'''




def remove_right_hand(input_position):
  """ Set all the coordinates for the right hand to 0

  Args:
    input_position: full body position
  Returns:
    position_wo_r_hand : position, where right hand is nulified
  """

  # Go over all the frames in the input
  for frame_id in range(FLAGS.amount_of_frames_as_input):
    offset = FLAGS.frame_size * frame_id
    coords_before_right_arm = input_position[:,:, 0 + offset : 18 + offset ]
    coords_after_right_arm = input_position[:,:, 30 + offset : 66 + offset]
    zeros_for_right_arm =  [[[0 for i in range(12)] for j in range(FLAGS.chunk_length)] for k in range(FLAGS.batch_size)]
    frame_wo_r_hand = tf.concat((coords_before_right_arm, zeros_for_right_arm, coords_after_right_arm), axis=2)
    # add next frame with r hand nullified to the resulting position
    if(frame_id) == 0:
      position_wo_r_hand = frame_wo_r_hand
    else:
      position_wo_r_hand = tf.concat((position_wo_r_hand, frame_wo_r_hand), axis = 2)

  return position_wo_r_hand


#if __name__ == '__main__':

