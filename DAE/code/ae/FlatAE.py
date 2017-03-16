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
      lstm_size = self.__shape[self.__recurrent_layer] if self.__recurrent_layer <= self.num_hidden_layers+1 else self.num_hidden_layers+1
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
      dropout = FLAGS.dropout # (keep probability) value
      variance = FLAGS.variance_of_noise
      batch_size = FLAGS.batch_size
      chunk_length = FLAGS.chunk_length

      #Define the network itself
      self._input_ = tf.placeholder(dtype=tf.float32,
                                    shape=(None, FLAGS.chunk_length, FLAGS.DoF), #FLAGS.batch_size
                                    name='ae_input_pl')
      self._target_ = tf.placeholder(dtype=tf.float32,
                                     shape=(None, FLAGS.chunk_length, FLAGS.DoF),
                                     name='ae_target_pl')

      # Define output and loss for the training data
      output = self.process_sequences(self._input_, dropout) # process batch of sequences
      self._loss = loss_reconstruction(output, self._target_) /(batch_size*chunk_length)

      # Define output and loss for the test data
      self._test_output = self.process_sequences(self._input_, 1) # we do not have dropout during testing
      with tf.name_scope("eval"):
        self._test_loss = loss_reconstruction(self._test_output, self._target_) /(batch_size*chunk_length)
        

      ##############        DEFINE OPERATIONS       ###############################################

    # Define optimizers
    learning_rate = FLAGS.training_learning_rate
    optimizer =  tf.train.RMSPropOptimizer(learning_rate=learning_rate) # GradientDescentOptimizer
        
    print('Optimizer was created')
        
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),   1e12)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars),  global_step = tf.contrib.framework.get_or_create_global_step())
    print('Training operator was created')

  def single_run(self, input_pl, time_step, just_middle = False):
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
                
            if(i==self.__recurrent_layer):
              if time_step > 0: tf.get_variable_scope().reuse_variables()
              (last_output, self._RNN_state) = self._RNN_cell(last_output, self._RNN_state)

            w = self._w(i + 1)
            b = self._b(i + 1)
            last_output = self._activate(last_output, w, b)

          # Maybe apply recurrency at the output layer
          if(self.num_hidden_layers+1==self.__recurrent_layer):
              if time_step > 0: tf.get_variable_scope().reuse_variables()
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
            
          # First - Apply Dropout
          the_whole_sequences = tf.nn.dropout(input_seq_pl, dropout)

          # Take batches for every time step and run them through the network
          # Stack all their outputs
          with tf.control_dependencies([tf.convert_to_tensor(self._RNN_state, name='state') ]): # do not let paralelize the loop
            stacked_outputs = tf.stack( [ self.single_run(the_whole_sequences[:,time_st,:], time_st, just_middle) for time_st in range(self.sequence_length) ])

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

  def read_process_write_bvh_file(self,input_seq_file_name, max_val, mean_pose, output_bvh_file_name):
   print('Take a test sequence from the file',input_seq_file_name)
   with self.session.graph.as_default():
    sess = self.session

    #                    GET THE DATA
        
    # get input sequnce
    inputSequence = read_file(input_seq_file_name)

    # Split it into chunks
    chunks = np.array([inputSequence [i:i + self.__sequence_length, :] for i in xrange(0, len(inputSequence )-self.__sequence_length + 1, FLAGS.chunking_stride)]) # Split sequence into chunks

    # Substract the mean pose
    chunks_minus_mean = chunks - mean_pose[np.newaxis,np.newaxis,:]

    # Scales all values in the input_data to be between -1 and 1
    eps=1e-15
    chunks_normalized =np.divide(chunks_minus_mean,max_val[np.newaxis,np.newaxis,:]+eps)

    # Batch those chunks
    batches = np.array([chunks_normalized[i:i + self.__batch_size, :] for i in xrange(0, len(chunks_normalized)-self.__batch_size + 1, FLAGS.chunking_stride)])

    numb_of_batches = batches.shape[0]

    #DEBUG
    ''' print('Batch size is ', self.__batch_size)
    print('We have got ',chunks_normalized.shape[0], 'sequences')
    print('Process it by AE as  ', numb_of_batches , ' batches') '''

    #                    RUN THE NETWORK

    # pass the batches of chunks through the AE
    output_batches= np.array( [ sess.run(self._test_output , feed_dict={self._input_: batches[i]}) for i in range(numb_of_batches)])

    # Unroll it to back to the sequence
    output_chunks = output_batches.reshape(-1, output_batches.shape[-1])

    # Convert it back from [-1,1] to original values
    reconstructed = np.multiply(output_chunks,max_val[np.newaxis,np.newaxis,:]+eps)
    
    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis,np.newaxis,:]

    #Unroll batches into the sequence
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])

    numb_of_chunks = reconstructed.shape[0]

    # Include rotations as well
    rotations = np.array( [  [0,0,0] for time_st  in range(numb_of_chunks)] ) #in range(self.__sequence_length) for snippet
    reconstructed = np.concatenate((reconstructed[:,0:3],rotations,reconstructed[:,3:]), axis=1)
    
    np.savetxt(output_bvh_file_name, reconstructed , fmt='%.5f', delimiter=' ')

   print('And write an output into the file ' + output_bvh_file_name + '...')

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
          """Get the output of the autoencoder

          Args:
            input_seq_pl:     tf placeholder for ae input data of size [batch_size, sequence_length, DoF]
            dropout:          how much of the input neurons will be activated, value in [0,1]
            just_middle :     will indicate if we want to extract only the middle layer of the network
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
