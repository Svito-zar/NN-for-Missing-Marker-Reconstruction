from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.contrib import rnn
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
    self.__recurrent_layer = FLAGS.recurrent_layer 

    self.__sequence_length = FLAGS.chunk_length
    self.__batch_size = FLAGS.batch_size

    self.__variables = {}
    self.__sess = sess

    with tf.variable_scope("AE_Variables"):

      ##############        SETUP VARIABLES       ###############################################
      
      # Create a variable to track the global step.
      global_step = tf.get_variable(name='global_step', shape=[1], initializer=tf.constant_initializer(0.0)) #tf.Variable(0, name='global_step', trainable=False)
      
      for i in xrange(self.__num_hidden_layers + 1):
            # Train weights
            name_w = self._weights_str.format(i + 1)
            w_shape = (self.__shape[i], self.__shape[i + 1])
            a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
            #w_init = tf.random_uniform(w_shape, -1 * a, a)
            self[name_w] = tf.get_variable(name_w,
                            initializer=tf.random_uniform(w_shape, -1 * a, a))
            # Train biases
            name_b = self._biases_str.format(i + 1)
            b_shape = (self.__shape[i + 1],)
            #b_init = tf.zeros(b_shape)
            self[name_b] = tf.get_variable(name_b, initializer=tf.zeros(b_shape))

      # Define LSTM cell
      lstm_size = self.__shape[self.__recurrent_layer] if self.__recurrent_layer < self.num_hidden_layers+1 else self.num_hidden_layers+1
      num_LSTM_layers = 1 # TODO: change
      def lstm_cell():
          return tf.contrib.rnn.BasicLSTMCell(
            lstm_size, forget_bias=1.0, state_is_tuple=True)
      self._cell = lstm_cell() #tf.contrib.rnn.MultiRNNCell(
                 # [lstm_cell() for _ in range(num_LSTM_layers)], state_is_tuple=True)
              
      self._initial_state = self._cell.zero_state(self.batch_size, tf.float32)


      ##############        DEFINE FUNCTIONS       ###############################################

      
      def single_run(self, input_pl, time_step, state, just_middle = False):
          """Get the output of the autoencoder for a single batch

          Args:
            input_pl:     tf placeholder for ae input data of size [batch_size, DoF]
            state:        current state of LSTM memory units
            just_middle : will indicate if we want to extract only the middle layer of the network
          Returns:
            Tensor of output
          """

          #Debug
          #state_val = self.__sess.run(state,feed_dict={lstm : self['lstm']})

          lstm = self._cell
          last_output = input_pl

          # print(time_step)

          # Pass through the network
          for i in xrange(self.num_hidden_layers+1):
                
            if(i==self.__recurrent_layer):
              if time_step > 0: tf.get_variable_scope().reuse_variables()
              (last_output, state) = lstm(last_output, state)

            w = self._w(i + 1)
            b = self._b(i + 1)
            last_output = self._activate(last_output, w, b)

          return last_output

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

          # Initial state of the LSTM memory.
          state = self._initial_state
            
          # First - Apply Dropout
          the_whole_sequences = tf.nn.dropout(input_seq_pl, dropout)

          # Take batches for every time step and run them through the network
          # Stack all their outputs
          with tf.control_dependencies([tf.convert_to_tensor(state, name='state') ]): # do not let paralelize the loop
            stacked_outputs = tf.stack( [ single_run(self,the_whole_sequences[:,time_st,:], time_st, state, just_middle) for time_st in range(self.sequence_length) ])

          # Transpose output from the shape [sequence_length, batch_size, DoF] into [batch_size, sequence_length, DoF]

          output = tf.transpose(stacked_outputs , perm=[1, 0, 2])

          # print('The final result has a shape:', output.shape)
          
          return output

      ##############        DEFINE THE NETWORK LOSS       ###############################################
        
      # Get some constants from the flags file
      keep_prob = tf.placeholder(tf.float32) #dropout placeholder
      dropout = FLAGS.dropout # (keep probability) value
      variance = FLAGS.variance_of_noise
      batch_size = FLAGS.batch_size
      chunk_length = FLAGS.chunk_length

      #Define the network itself
      self._input_ = tf.placeholder(dtype=tf.float32,
                                    shape=(FLAGS.batch_size, FLAGS.chunk_length, FLAGS.DoF),
                                    name='ae_input_pl')
      self._target_ = tf.placeholder(dtype=tf.float32,
                                     shape=(FLAGS.batch_size, FLAGS.chunk_length, FLAGS.DoF),
                                     name='ae_target_pl')

      # Define output and loss for the training data
      output = process_sequences(self,self._input_, dropout) # process batch of sequences
      self._loss = tf.nn.l2_loss(tf.subtract(output, self._target_)) /(batch_size*chunk_length)

      # Define output and loss for the test data
      test_output = process_sequences(self, self._input_, 1) # we do not have dropout during testing
      with tf.name_scope("eval"):
        self._test_loss = tf.nn.l2_loss(tf.subtract(test_output, self._target_)) /(batch_size*chunk_length)
        

      ##############        DEFINE OPERATIONS       ###############################################

    # Define optimizers
    learning_rate = FLAGS.pretraining_learning_rate
    optimizer =  tf.train.RMSPropOptimizer(learning_rate=learning_rate) # GradientDescentOptimizer
        

    print('Optimizer was created')
        
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),   1e12)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars),  global_step = tf.contrib.framework.get_or_create_global_step())
    print('Training operator was created')
        
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
