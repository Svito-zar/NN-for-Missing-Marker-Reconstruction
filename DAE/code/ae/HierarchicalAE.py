from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import scipy.io as sio
import tensorflow as tf
import time
from utils.data import fill_feed_dict_ae, read_unlabeled_data, read_file, loss_reconstruction
from utils.flags import FLAGS
from utils.utils import tile_raster_images


class HierarchicalAE(object):
  """Hierarchical deep autoencoder.

  Deep Autoencoder with Hierarchical encoding, used for full training cycle.
  The user specifies the structure of the neural net
  by specifying number of inputs and the number of hidden
  units for each layer corresponding to each body part.
  """
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  # The following two array must have the same size and ordering, they will be use to make a proper connection for the input layer
  _body_part_names = ["spine", "r_arm", "l_arm", "r_leg", "l_leg"]
  _input_shapes = [18, 45, 45, 12, 12]      # we don't use the information about the whole body for each body part, but add it directly to the represantation layer
  _body_channels = 3                        # DoF affecting the whole body
  _output_shapes = [18, 42, 42, 12, 12]     # Do not duplicate

  def __init__(self, DoF, encode1, encode2, encode3, sess):
    """Autoencoder initializer

    Args:
      DoF:        number of Degrees Of Freedom in the input data
      encode1 :   list of integers, specifiyng number of neurons for each body part in the first hidden layer,
                  n1_head_chest, n1_rArm, n1_lArm, n1_rLeg, n1_lLeg
      encode2 :   list of integers, specifiyng number of neurons for the upper body and the lower body
                  n3_upper, n3_lower
      encode3 :   integer specifying the size of the 3rd layer, containing information about the whole body
                  it will be a middle layer for representation
      decode  :   is simetric to encoding
      sess: tensorflow session object to use
    """
    self.__DoF = DoF
    self.__encode1 = encode1
    self.__encode2 = encode2
    self.__encode3 = encode3
    self.__shape = [DoF, encode1.sum(), encode2.sum(), encode3.sum(), encode2.sum(), encode1.sum(), DoF]

    self.__num_hidden_layers = 5 # HARDCODED NUMBER
    self.__recurrent_layer = FLAGS.recurrent_layer
    self.__sequence_length = FLAGS.chunk_length
    self.__batch_size = FLAGS.batch_size                         # The size of the current batch AE is working with

    self.__variables = {}
    self.__sess = sess

    self._setup_variables()

  @property
  def DoF(self):
    return self.__DoF

  @property
  def batch_size(self):
    return self.__batch_size

  @property
  def sequence_length(self):
    return self.__sequence_length
  
  @property
  def num_hidden_layers(self):
    return self.__num_hidden_layers

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
    with tf.variable_scope("hiererchical_autoencoder_variables"):

      print('Setting up the variables for Hierarchical Autoencoder...\n')

      ######################            Encoding          #############################
      
      # *****************          First layer weights and biases**********
      
      for bp_id in range(len(self.__encode1)): # bp_id - body part ID
        name_of_part = self._body_part_names[bp_id]
        # matrices
        name_w = "w_" + name_of_part
        w_shape = (self._input_shapes[bp_id], self.__encode1[bp_id])
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                     name=name_w,
                                     trainable=True)

        #DEBUG
        # print('Next body part name is : ', name_of_part,' Its matrix has a shape : ', w_shape )
      
        # biases
        name_b = "b_" + name_of_part
        b_shape = (self.__encode1[bp_id])
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

      
      # *******************      Second layer weights and biases   ************
      name_w = "w_upper_body"
      w_shape = (self.__encode1[0] + self.__encode1[1] + self.__encode1[2], self.__encode2[0]) # spine and both arms -> upper body
      a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
      w_init = tf.random_uniform(w_shape, -1 * a, a)
      self[name_w] = tf.Variable(w_init,
                                  name=name_w,
                                  trainable=True)
      name_b = "b_upper_body"
      b_shape = (self.__encode2[0])
      b_init = tf.zeros(b_shape)
      self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)
      #DEBUG
      #print('\nNext body part name is : ', name_w , 'It has a shape : ', w_shape )
      
      name_w = "w_lower_body"  
      w_shape = (self.__encode1[3] + self.__encode1[4], self.__encode2[1]) # both legs -> lower body
      a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
      w_init = tf.random_uniform(w_shape, -1 * a, a)
      self[name_w] = tf.Variable(w_init,
                                  name=name_w,
                                  trainable=True)
      name_b = "b_lower_body"
      b_shape = (self.__encode2[1])
      b_init = tf.zeros(b_shape)
      self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)
      #DEBUG
      #print('Next body part name is : ', name_w , 'It has a shape : ', w_shape )

      #      ****************  Third layer weight and bias   ***************
      
      name_w = "w_whole_body"
        
      w_shape = (self.__encode2[0]+self.__encode2[1] + self._body_channels, int(self.__encode3))
      a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
      w_init = tf.random_uniform(w_shape, -1 * a, a)
      self[name_w] = tf.Variable(w_init,
                                      name=name_w,
                                      trainable=True)
      name_b = "b_whole_body"
      b_shape = (self.__encode3)
      b_init = tf.zeros(b_shape)
      self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)
      
      #DEBUG
      #print('\nMatrix before the whole body representation has a shape : ', w_shape )
        

      ######################            Decoding          #############################
      
      #  ************      First decoding layer weights and biases   **********
      name_w = "w_whole_body_decode"  
      w_shape = (int(self.__encode3), self.__encode2[0]+self.__encode2[1] + self._body_channels)
      a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
      w_init = tf.random_uniform(w_shape, -1 * a, a)
      self[name_w] = tf.Variable(w_init,
                                      name=name_w,
                                      trainable=True)
      # biase
      name_b = "b_whole_body_decode"
      b_shape = (self.__encode2[0]+self.__encode2[1] + self._body_channels)
      b_init = tf.zeros(b_shape)
      self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)


      # *************       Second decoding layer weights and biases     *************
     
      upper_body_in_size = self.__encode1[0] + self.__encode1[1] + self.__encode1[2]
      lower_body_in_size = self.__encode1[3] + self.__encode1[4]
      
      sizes = [upper_body_in_size, lower_body_in_size]
      ind = 0
      for name_of_part in ['upper_body', 'lower_body']:
        name_w = "w_" + name_of_part + "_decode"
        
        w_shape = (self.__encode2[ind], sizes[ind] )
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                       name=name_w,
                                       trainable=True)
        name_b = "b_" + name_of_part+ "_decode"
        b_shape = (sizes[ind])
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

        ind+= 1

      
      # Third decoding layer weights and biases
      for bp_id in range(len(self.__encode1)): # bp_id - body part ID
        name_of_part = self._body_part_names[bp_id]
        # matrices
        name_w = "w_" + name_of_part + "_decode"
        w_shape = ( self.__encode1[bp_id], self._output_shapes[bp_id]) # NOTE usage of output_shapes instead of input_shapes
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                     name=name_w,
                                     trainable=True)
        # biases
        name_b = "b_" + name_of_part + "_decode"
        b_shape = (self._output_shapes[bp_id])
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

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

      self._reconstruction_loss =  loss_reconstruction(output, self._target_) /(batch_size*chunk_length)
      tf.add_to_collection('losses', self._reconstruction_loss)
      self._loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

      # Define output and loss for the test data
      self._test_output = self.process_sequences(self._input_, 1) # we do not have dropout during testing
      with tf.name_scope("eval"):
        self._test_loss = loss_reconstruction(self._test_output, self._target_) /(batch_size*chunk_length)        

    
  def _w(self, n, suffix=""):
    return self["w_" + suffix]

  def _b(self, n, suffix=""):
    return self["b_" + suffix]

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def single_run(self, input_pl, time_step, dropout, just_middle = False):
      """Get the output of the autoencoder

      Args:
        input_pl: tf placeholder for a current batch
        dropout:  how much of the input neurons will be activated, value in [0,1]
        just_middle : will indicate if we want to extract only the middle layer of the network
      Returns:
        Tensor of output
      """
      with tf.name_scope("hiererchical_AE_run"):

        # First - Apply Dropout
        last_output = tf.nn.dropout(input_pl, dropout)
        
        #########################            ENCODING                ######################3

        #Extract the part of the data, which belongs to each body part
        # This part is format specific. And hardcoded!
        # The data for the structure of the MoCap dataset (in BVH format) is in the file BVH_fierarchy in the root folder of my Dropbox folder
        with tf.variable_scope("Encoding"):
          
          with tf.variable_scope("Body_part_extraction"):
            
            indices = [ [ [elem, index] for index in range(3) ] for elem in range(self.__batch_size)]
            body_in = tf.gather_nd(last_output, indices, name=None) # changes the whole body
            
            indices = [ [ [elem, index] for index in range(3,21,1) ] for elem in range(self.__batch_size)]
            spine = tf.gather_nd(last_output, indices, name=None) # chest and head
        
            indices = [ [ [elem, index] for index in range(6,9,1) ] + [ [elem, index] for index in range(21,63,1) ]  for elem in range(self.__batch_size)]
            r_Arm = tf.gather_nd(last_output, indices, name=None) # right arm

            indices = [ [ [elem, index] for index in range(6,9,1) ] + [ [elem, index] for index in range(63,105,1) ]  for elem in range(self.__batch_size)]
            l_Arm = tf.gather_nd(last_output, indices, name=None) # left arm

            indices = [ [ [elem, index] for index in range(105,117,1) ] for elem in range(self.__batch_size)]
            r_Leg = tf.gather_nd(last_output, indices, name=None) # right leg

            indices = [ [ [elem, index] for index in range(117,129,1) ] for elem in range(self.__batch_size)]
            l_Leg = tf.gather_nd(last_output, indices, name=None) # left leg

            hierarchical_input = [spine, r_Arm, l_Arm, r_Leg, l_Leg]
          
          # Then pass through the network

          ################    First layer encoding    #################

          with tf.variable_scope("Body_part_encoding"):
            
            first_hidden_layer = []
            for bp_id in range(len(self.__encode1)): # bp_id - body part ID
              name_of_body_part = self._body_part_names[bp_id]
              first_hidden_layer.append(self._activate(hierarchical_input[bp_id], self["w_"+name_of_body_part], self["b_"+name_of_body_part]))


          with tf.variable_scope("Upper_lower_concat"):
            
            # Combine the outputs
            upper_body_in = tf.concat([first_hidden_layer[0], first_hidden_layer[1], first_hidden_layer[2]], 1)
            lower_body_in = tf.concat([first_hidden_layer[3], first_hidden_layer[4]], 1)

          
          ################    Second layer encoding    #################

          # Apply Dropout
          upper_body_in = tf.nn.dropout(upper_body_in, dropout)
          lower_body_in = tf.nn.dropout(lower_body_in, dropout)
        
          with tf.variable_scope("Upper_lower_encoding"):
            
            ################    Thirsd layer encoding    #################
            upper_body_out = self._activate(upper_body_in, self["w_upper_body"], self["b_upper_body"])
            lower_body_out = self._activate(lower_body_in, self["w_lower_body"], self["b_lower_body"])
          
          with tf.variable_scope("Whole_body_concat"):
            # Concatanate
            whole_body_in = tf.concat([body_in, upper_body_out, lower_body_out],1,name = 'whole_body_in')

          ################    Third layer encoding    ##################

          # Apply Dropout
          whole_body_in = tf.nn.dropout(whole_body_in, dropout)
          with tf.variable_scope("Last_hiddeb_layer"):
            representation = self._activate(whole_body_in, self["w_whole_body"], self["b_whole_body"])

          if(just_middle):
            return representation

        #########################            DECODING                ######################3

        with tf.variable_scope("Decoding"):
            ################    1st layer decoding    ##################

          # Apply Dropout
          representation = tf.nn.dropout(representation, dropout)
          
          with tf.variable_scope("Whole_body_decode"):
            whole_body_decode = self._activate(representation , self["w_whole_body_decode"], self["b_whole_body_decode"])

          # Slice it back into 3 parts : upper, lower body and the bits, which are affection the whole body
          with tf.variable_scope("Slice_into_upper_and_lower"):
            body_decode = tf.slice(whole_body_decode, [0,0], [self.__batch_size,self._body_channels], name = 'body_decode')
            upper_body_slice = tf.slice(whole_body_decode, [0,self._body_channels], [self.__batch_size,FLAGS.upper_body_neurons], name = 'upper_body_slice')
            lower_body_slice = tf.slice(whole_body_decode, [0,self._body_channels+FLAGS.upper_body_neurons], [self.__batch_size,FLAGS.lower_body_neurons], name = 'lower_body_slice')

          with tf.variable_scope("Decode_upper_and_lower"):
            
            ################    2nd layer decoding    ##################

            # Apply Dropout
            upper_body_slice  = tf.nn.dropout(upper_body_slice, dropout)
            lower_body_slice  = tf.nn.dropout(lower_body_slice, dropout)
          
            upper_body_decode = self._activate(upper_body_slice, self["w_upper_body_decode"], self["b_upper_body_decode"])
            lower_body_decode = self._activate(lower_body_slice, self["w_lower_body_decode"], self["b_lower_body_decode"])


          # Slice it back to the body parts
          with tf.variable_scope("Slice_into_body_parts"):
            spine_slice = tf.slice(upper_body_decode,[0,0],[self.__batch_size, self.__encode1[0]], name = 'spine_decode_slice')
            r_arm_slice = tf.slice(upper_body_decode,[0,self.__encode1[0]],[self.__batch_size, self.__encode1[1]])
            l_arm_slice = tf.slice(upper_body_decode,[0,self.__encode1[0]+self.__encode1[1]],[self.__batch_size, self.__encode1[2]])
            r_leg_slice = tf.slice(lower_body_decode,[0,0],[self.__batch_size, self.__encode1[3]])
            l_leg_slice = tf.slice(lower_body_decode,[0,self.__encode1[3]],[self.__batch_size, self.__encode1[4]])
            

          ################    4rd decoding  layer  ##################
          with tf.variable_scope("Decode_body_parts"):
            r_arm_decode = self._activate(tf.nn.dropout(r_arm_slice, dropout), self["w_r_arm_decode"], self["b_r_arm_decode"])
            l_arm_decode = self._activate(tf.nn.dropout(l_arm_slice, dropout), self["w_l_arm_decode"], self["b_l_arm_decode"])
            r_leg_decode = self._activate(tf.nn.dropout(r_leg_slice, dropout), self["w_r_leg_decode"], self["b_r_leg_decode"])
            l_leg_decode = self._activate(tf.nn.dropout(l_leg_slice, dropout), self["w_l_leg_decode"], self["b_l_leg_decode"])
            spine_decode = self._activate(tf.nn.dropout(spine_slice, dropout), self["w_spine_decode"], self["b_spine_decode"])

          with tf.variable_scope("combine"):
            output = tf.concat([body_decode, spine_decode, r_arm_decode, l_arm_decode,r_leg_decode,l_leg_decode], 1, name='concat')

      return output

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
      stacked_outputs = tf.stack( [ self.single_run(input_seq_pl[:,time_st,:], time_st, dropout, just_middle) for time_st in range(self.sequence_length) ])

    # Transpose output from the shape [sequence_length, batch_size, DoF] into [batch_size, sequence_length, DoF]

    output = tf.transpose(stacked_outputs , perm=[1, 0, 2])

    #print('The final result has a shape:', output.shape)
          
    return output
        
  def single_run_shallow(self, input_pl):
      with tf.name_scope("shallow_run"):

        last_output = input_pl
        
        with tf.variable_scope("Shalow_Encoding"):
            with tf.variable_scope("Extract_Body_parts"):
              indices = [ [ [elem, index] for index in range(3) ] for elem in range(self.__batch_size)]
              body_in = tf.gather_nd(last_output, indices, name=None) # changes the whole body
              
              indices = [ [ [elem, index] for index in range(3,21,1) ] for elem in range(self.__batch_size)]
              spine = tf.gather_nd(last_output, indices, name=None) # chest and head
          
              indices = [ [ [elem, index] for index in range(6,9,1) ] + [ [elem, index] for index in range(21,63,1) ]  for elem in range(self.__batch_size)]
              r_Arm = tf.gather_nd(last_output, indices, name=None) # right arm

              indices = [ [ [elem, index] for index in range(6,9,1) ] + [ [elem, index] for index in range(63,105,1) ]  for elem in range(self.__batch_size)]
              l_Arm = tf.gather_nd(last_output, indices, name=None) # left arm

              indices = [ [ [elem, index] for index in range(105,117,1) ] for elem in range(self.__batch_size)]
              r_Leg = tf.gather_nd(last_output, indices, name=None) # right leg

              indices = [ [ [elem, index] for index in range(117,129,1) ] for elem in range(self.__batch_size)]
              l_Leg = tf.gather_nd(last_output, indices, name=None) # left leg

              hierarchical_input = [spine, r_Arm, l_Arm, r_Leg, l_Leg]

            with tf.variable_scope("Encode_Body_parts"):
              spine_repres   = self._activate(spine, self["w_spine"], self["b_spine"])
              r_arm_repres = self._activate(r_Arm, self["w_r_arm"], self["b_r_arm"])
              l_arm_repres = self._activate(l_Arm, self["w_l_arm"], self["b_l_arm"])
              r_leg_repres = self._activate(r_Leg, self["w_r_leg"], self["b_r_leg"])
              l_leg_repres = self._activate(l_Leg, self["w_l_leg"], self["b_l_leg"])
              
            
                
        with tf.variable_scope("Shalow_Decoding"):
            with tf.variable_scope("Decode_body_parts"):
              r_arm_decode = self._activate(r_arm_repres, self["w_r_arm_decode"], self["b_r_arm_decode"])
              l_arm_decode = self._activate(l_arm_repres, self["w_l_arm_decode"], self["b_l_arm_decode"])
              r_leg_decode = self._activate(r_leg_repres, self["w_r_leg_decode"], self["b_r_leg_decode"])
              l_leg_decode = self._activate(l_leg_repres, self["w_l_leg_decode"], self["b_l_leg_decode"])
              spine_decode = self._activate(spine_repres, self["w_spine_decode"], self["b_spine_decode"])

            with tf.variable_scope("concatanate"):
              output = tf.concat([body_in, spine_decode, r_arm_decode, l_arm_decode,r_leg_decode,l_leg_decode],1, name='concat')
              
      return output

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
          stacked_outputs = tf.stack( [ self.single_run_shallow(the_whole_sequences[:,time_st,:]) for time_st in range(self.sequence_length) ])

          # Transpose output from the shape [sequence_length, batch_size, DoF] into [batch_size, sequence_length, DoF]

          output = tf.transpose(stacked_outputs , perm=[1, 0, 2])

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
      self.__batch_size = len(currSequence)
      print("Length of curr seq. : " + str(self.__batch_size))

      # Define on an operator
      middle_op = self.run_net(input_ , 1, just_middle = True) # 1 means that we have no dropout
        
      # for each snippet in a sequence
      # pass through the network untill the middle layer
      middle = sess.run(middle_op, feed_dict={input_: currSequence})
        
      # save it into a file
      sio.savemat(output_seq_file_name, {'trialId':name, 'spikes':np.transpose(middle)})  
