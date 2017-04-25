from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.contrib import rnn
import time
from utils.data import fill_feed_dict_ae, read_unlabeled_data, read_file, loss_reconstruction
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

  def __init__(self, shape, sess, learning_rate, batch_size, dropout, variance):
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
    self.__batch_size = batch_size

    self.__variables = {}
    self.__sess = sess

    if(FLAGS.Weight_decay is not None):
      print('We apply weight decay')

    with tf.variable_scope("AE_Variables"):

      ##############        SETUP VARIABLES       ###############################################
      
      # Create a variable to track the global step.
      global_step = tf.get_variable(name='global_step', shape=[1], initializer=tf.constant_initializer(0.0)) #tf.Variable(0, name='global_step', trainable=False)
      
      for i in xrange(self.__num_hidden_layers + 1): # go over all layers

        self._create_variables(i, FLAGS.Weight_decay)

      # Define LSTM cell
      lstm_size = self.__shape[self.__recurrent_layer+1] if self.__recurrent_layer <= self.num_hidden_layers+1 else self.num_hidden_layers+1
      num_LSTM_layers = 1 # TODO: change
      def lstm_cell():
          return tf.contrib.rnn.BasicLSTMCell(
            lstm_size, forget_bias=1.0, state_is_tuple=True)
      self._RNN_cell = lstm_cell() #tf.contrib.rnn.MultiRNNCell(
                 # [lstm_cell() for _ in range(num_LSTM_layers)], state_is_tuple=True)
              
      self._initial_state = self._RNN_cell.zero_state(self.batch_size, tf.float32)

      ##############        DEFINE THE NETWORK LOSS       ###############################################
        
      # Get some constants from the flags file
      keep_prob = tf.placeholder(tf.float32) #dropout placeholder
      chunk_length = self.sequence_length

      #Define the network itself
      self._input_ = tf.placeholder(dtype=tf.float32,
                                    shape=(None, chunk_length, FLAGS.DoF), #FLAGS.batch_size
                                    name='ae_input_pl')
      self._target_ = tf.placeholder(dtype=tf.float32,
                                     shape=(None, chunk_length, FLAGS.DoF),
                                     name='ae_target_pl')

      # Define output and loss for the training data
      output = self.process_sequences(self._input_, dropout) # process batch of sequences

      self._reconstruction_loss =  loss_reconstruction(output, self._target_) /(batch_size*chunk_length)
      tf.add_to_collection('losses', self._reconstruction_loss)
      self._loss =       tf.add_n(tf.get_collection('losses'), name='total_loss')

      # Define output and loss for the test data
      self._test_output = self.process_sequences(self._input_, 1) # we do not have dropout during testing
      with tf.name_scope("eval"):
        self._test_loss = loss_reconstruction(self._test_output, self._target_) /(batch_size*chunk_length)
        

  def single_run(self, input_pl, time_step, dropout, just_middle = False):
          """Get the output of the autoencoder for a single batch

          Args:
            input_pl:     tf placeholder for ae input data of size [batch_size, DoF]
            state:        current state of LSTM memory units
            just_middle : will indicate if we want to extract only the middle layer of the network
          Returns:
            Tensor of output
          """
          #print(self._RNN_state)
	
	  last_output = input_pl

          # Pass through the network
          for i in xrange(self.num_hidden_layers+1):

                
            if(i!=self.__recurrent_layer):
              w = self._w(i + 1)
              b = self._b(i + 1)
              last_output = self._activate(last_output, w, b)

            else:
              if time_step > 0:
                tf.get_variable_scope().reuse_variables()
              (last_output, self._RNN_state) = self._RNN_cell(last_output, self._RNN_state)

          # Maybe apply recurrency at the output layer
          if(self.num_hidden_layers+1==self.__recurrent_layer):
              if time_step > 0:
                tf.get_variable_scope().reuse_variables()
              (last_output, self._RNN_state) = self._RNN_cell(last_output, self._RNN_state)

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
          self._RNN_state = self._initial_state
            
          # Take batches for every time step and run them through the network
          # Stack all their outputs
          with tf.control_dependencies([tf.convert_to_tensor(self._RNN_state, name='state') ]): # do not let paralelize the loop
            stacked_outputs = tf.stack( [ self.single_run(input_seq_pl[:,time_st,:], time_st,dropout, just_middle) for time_st in range(self.sequence_length) ])

          # Transpose output from the shape [sequence_length, batch_size, DoF] into [batch_size, sequence_length, DoF]

          output = tf.transpose(stacked_outputs , perm=[1, 0, 2])

          # print('The final result has a shape:', output.shape)
          
          return output
        
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
        w = self._w(self.__num_hidden_layers+1)
        b = self._b(self.__num_hidden_layers+1)
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

    if i < self.__num_hidden_layers:
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
    assert n <= self.__num_hidden_layers

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

