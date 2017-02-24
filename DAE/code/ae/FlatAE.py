from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import scipy.io as sio
import tensorflow as tf
import time
from utils.data import fill_feed_dict_ae, read_unlabeled_data, read_file
from utils.flags import FLAGS


class FlatAutoEncoder(object):
  """Generic deep autoencoder.

  Autoencoder used for full training cycle.
  The user specifies the structure of the neural net
  by specifying number of inputs, the number of hidden
  units for each layer and the number of final outputs.
  """
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  def __init__(self, shape, sess):
    """Autoencoder initializer

    Args:
      shape: list of ints specifying
              num input, hidden1 units,...hidden_n units, num logits
      sess: tensorflow session object to use
    """
    self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
    self.__num_hidden_layers = len(self.__shape) - 2

    self.__sequence_length = FLAGS.chunk_length
    self.__batch_size = FLAGS.batch_size

    self.__variables = {}
    self.__sess = sess

    self._setup_variables()

  @property
  def shape(self):
    return self.__shape
  
  @property
  def num_hidden_layers(self):
    return self.__num_hidden_layers

  @property
  def sequence_length(self):
    return self.__sequence_length

  @property
  def batch_size(self):
    return self.__batch_size

  @property
  def session(self):
    return self.__sess

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

  def _setup_variables(self):
    with tf.name_scope("autoencoder_variables"):
      for i in xrange(self.__num_hidden_layers + 1):
        # Train weights
        name_w = self._weights_str.format(i + 1)
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                   name=name_w,
                                   trainable=True)
        # Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

        if i <= self.__num_hidden_layers:

          # Pretraining output training biases
          name_b_out = self._biases_str.format(i + 1) + "_out"
          b_shape = (self.__shape[i],)
          b_init = tf.zeros(b_shape)
          self[name_b_out] = tf.Variable(b_init,
                                         trainable=True,
                                         name=name_b_out)

  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n) + suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n) + suffix]

  def get_variables(self):
    """Return list of all variables of Autoencoder
    """

    vars=[]

    for n in range(self.__num_hidden_layers):
      vars.append(self._w(n+1))
      vars.append(self._b(n+1))

    return vars

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b)) # was sigmoid before
    return y

  def process_sequences(self, input_seq_pl, dropout, just_middle = False):
      """Get the output of the autoencoder

      Args:
        input_seq_pl:     tf placeholder for ae input data of size [batch_size, sequence_length, DoF]
        dropout:          how much of the input neurons will be activated, value in [0,1]
        just_middle :     will indicate if we want to extract only the middle layer of the network
      Returns:
        Tensor of output
      """

      if(~just_middle): # if not middle layer
        numb_layers = self.__num_hidden_layers+1
      else:
        numb_layers = FLAGS.middle_layer
      
      # First - Apply Dropout
      the_whole_sequences = tf.nn.dropout(input_seq_pl, dropout)

      #output = np.empty([self.sequence_length, self.batch_size, self.shape[0]])

      # Take batches for every time step and run them through the network
      # Stack all their outputs
      stacked_outputs = tf.stack([self.single_run(the_whole_sequences[:,time_st,:], just_middle) for time_st in range(self.sequence_length)])

      # Transpose output from the shape [sequence_length, batch_size, DoF] into [batch_size, sequence_length, DoF]

      output = tf.transpose(stacked_outputs , perm=[1, 0, 2])

      #print('The final result has a shape:', output.shape)
      
      return output

      
  def single_run(self, input_pl, just_middle = False):
      """Get the output of the autoencoder for a single batch

      Args:
        input_pl:     tf placeholder for ae input data of size [batch_size, DoF]
        just_middle : will indicate if we want to extract only the middle layer of the network
      Returns:
        Tensor of output
      """

      last_output = input_pl
      
      # Pass through the network
      for i in xrange(self.num_hidden_layers+1):
        w = self._w(i + 1)
        b = self._b(i + 1)

        last_output = self._activate(last_output, w, b)

      return last_output
    
  def write_middle_layer(self, input_seq_file_name, output_seq_file_name, name):
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
      currSequence = read_file(input_seq_file_name)

      # define tensors
      input_ = tf.placeholder(dtype=tf.float32,
                                    shape=(None, FLAGS.DoF),
                                    name='ae_input_pl')
      # Define the size of current input sequence
      self.__curr_batch_size = sess.run(tf.shape(input_ )[0], feed_dict={input_ : currSequence})

      # Define on an operator
      middle_op = self.run_net(input_ , 1, just_middle = True) # 1 means that we have no dropout
        
      # for each snippet in a sequence
      # pass through the network untill the middle layer
      middle = sess.run(middle_op, feed_dict={input_: currSequence})
        
      # save it into a file
      sio.savemat(output_seq_file_name, {'trialId':name, 'spikes':np.transpose(middle)})  
