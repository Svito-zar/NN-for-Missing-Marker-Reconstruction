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
from utils.eval import loss_supervised, evaluation, do_eval_summary
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
  _body_part_names = ["chest_head", "r_arm", "l_arm", "r_leg", "l_leg"]
  _input_shapes = [18, 45, 45, 12, 12]      # we don't use the information about the whole body for each body part, but add it directly to the represantation layer
  _body_channels = 3                        # DoF affecting the whole body
  _output_shapes = [18, 42, 42, 12, 12]     # Do not duplicate

  def __init__(self, DoF, encode1, encode2, encode3, encode4, sess):
    """Autoencoder initializer

    Args:
      DoF:        number of Degrees Of Freedom in the input data
      encode1 :   list of integers, specifiyng number of neurons for each body part in the first hidden layer,
                  n1_head_chest, n1_rArm, n1_lArm, n1_rLeg, n1_lLeg
      encode2 :   list of integers, specifiyng number of neurons for each body part in the second hidden layer,
                  n2_spine_rArm, n2_spine_lArm, n2_spine_rLeg, n2_spine_lLeg
      encode3 :   list of integers, specifiyng number of neurons for the upper body and the lower body
                  n3_upper, n3_lower
      encode4 :   integer specifying the size of the 3rd layer, containing information about the whole body
                  it will be a middle layer for representation
      decode  :   list of integers for decoding 
      sess: tensorflow session object to use
    """
    self.__DoF = DoF
    self.__encode1 = encode1
    self.__encode2 = encode2
    self.__encode3 = encode3
    self.__encode4 = encode4

    #self.__num_hidden_layers =  len(self.__decode) + 3                #  3 layers of encoding

    self.__curr_batch_size = FLAGS.batch_size                         # The size of the current batch AE is working with

    self.__variables = {}
    self.__sess = sess

    self._setup_variables()

  @property
  def DoF(self):
    return self.__DoF

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
    with tf.name_scope("hiererchical_autoencoder_variables"):

      print('Setting up the variables for Hierarchical Autoencoder...\n')

      ######################            Encoding          #############################
      
      # First layer weights and biases
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
        print('Next body part name is : ', name_of_part,' Its matrix has a shape : ', w_shape )
      
        # biases
        name_b = "b_" + name_of_part
        b_shape = (self.__encode1[bp_id])
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

      print('\n')
      
      # Second layer weights and biases
      for bp_id in range(len(self.__encode2)): # bp_id - body part ID
        name_of_part = "spine_"+self._body_part_names[bp_id+1] # skip head and chest
        # matrices
        name_w = "w_" + name_of_part
        w_shape = (self.__encode1[bp_id+1] +self.__encode1[0] , self.__encode2[bp_id])
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                     name=name_w,
                                     trainable=True)
        # biases
        name_b = "b_" + name_of_part
        b_shape = (self.__encode2[bp_id])
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)
        #DEBUG
        print('Next body part name is : ', name_of_part, 'It has a shape : ', w_shape )
      print('\n')
      
      # Third layer weights and biases
      ind = 0
      upper_body_in_size = self.__encode2[0] + self.__encode2[1]
      lower_body_in_size = self.__encode2[2] + self.__encode2[3]
      sizes = [upper_body_in_size, lower_body_in_size]
      for name_of_part in ['upper_body', 'lower_body']:
        name_w = "w_" + name_of_part
        
        w_shape = (sizes[ind] , self.__encode3[ind])
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                       name=name_w,
                                       trainable=True)
        #DEBUG
        print('Next body part name is : ', name_of_part, 'It has a shape : ', w_shape )
        
        name_b = "b_" + name_of_part
        b_shape = (self.__encode3[ind])
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

        ind+= 1

      # Forth layer weight and bias
      name_w = "w_whole_body"
        
      w_shape = (self.__encode3[0]+self.__encode3[1] + self._body_channels, int(self.__encode4))
      a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
      w_init = tf.random_uniform(w_shape, -1 * a, a)
      self[name_w] = tf.Variable(w_init,
                                      name=name_w,
                                      trainable=True)
      #DEBUG
      print('\nMatrix before the whole body representation has a shape : ', w_shape )
        
      # biase
      name_b = "b_whole_body"
      b_shape = (self.__encode4)
      b_init = tf.zeros(b_shape)
      self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

      ######################            Decoding          #############################
      
      # First decoding layer weights and biases
      name_w = "w_whole_body_decode"  
      w_shape = (int(self.__encode4), self.__encode3[0]+self.__encode3[1] + self._body_channels)
      a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
      w_init = tf.random_uniform(w_shape, -1 * a, a)
      self[name_w] = tf.Variable(w_init,
                                      name=name_w,
                                      trainable=True)
      # biase
      name_b = "b_whole_body_decode"
      b_shape = (self.__encode3[0]+self.__encode3[1] + self._body_channels)
      b_init = tf.zeros(b_shape)
      self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)


      # Second decoding layer weights and biases
      ind = 0
     
       #That was wrong! You need to fix this
      upper_body_in_size = self.__encode1[0] + self.__encode1[1] + self.__encode1[2]
      lower_body_in_size = self.__encode1[3] + self.__encode1[4]
      
      #upper_body_in_size = self.__encode2[0] + self.__encode2[1]
      #lower_body_in_size = self.__encode2[2] + self.__encode2[3]
      
      sizes = [upper_body_in_size, lower_body_in_size]
      for name_of_part in ['upper_body', 'lower_body']:
        name_w = "w_" + name_of_part + "_decode"
        
        w_shape = (self.__encode3[ind], sizes[ind] )
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

  def _w(self, n, suffix=""):
    return self["w_" + suffix]

  def _b(self, n, suffix=""):
    return self["b_" + suffix]

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def run_net(self, input_pl, dropout, just_middle = False):
      """Get the output of the autoencoder

      Args:
        numb_layers:    untill which layer should network run
        input_pl: tf placeholder for ae input data
        dropout:  how much of the input neurons will be activated, value in [0,1]
        just_middle : will indicate if we want to extract only the middle layer of the network
      Returns:
        Tensor of output
      """
      with tf.name_scope("hiererchical_AE_run"):
        #First - Apply Dropout
        last_output = tf.nn.dropout(input_pl, dropout)

        #########################            ENCODING                ######################3


        #Extract the part of the data, which belongs to each body part
        # This part is format specific. And hardcoded!
        # The data for the structure of the MoCap dataset (in BVH format) is in the file BVH_fierarchy in the root folder of my Dropbox folder
        indices = [ [ [elem, index] for index in range(3) ] for elem in range(self.__curr_batch_size)]
        body_in = tf.gather_nd(last_output, indices, name=None) # changes the whole body
        
        indices = [ [ [elem, index] for index in range(3,21,1) ] for elem in range(self.__curr_batch_size)]
        chest_head = tf.gather_nd(last_output, indices, name=None) # chest and head
    
        indices = [ [ [elem, index] for index in range(6,9,1) ] + [ [elem, index] for index in range(21,63,1) ]  for elem in range(self.__curr_batch_size)]
        r_Arm = tf.gather_nd(last_output, indices, name=None) # right arm

        indices = [ [ [elem, index] for index in range(6,9,1) ] + [ [elem, index] for index in range(63,105,1) ]  for elem in range(self.__curr_batch_size)]
        l_Arm = tf.gather_nd(last_output, indices, name=None) # left arm

        indices = [ [ [elem, index] for index in range(105,117,1) ] for elem in range(self.__curr_batch_size)]
        r_Leg = tf.gather_nd(last_output, indices, name=None) # right leg

        indices = [ [ [elem, index] for index in range(117,129,1) ] for elem in range(self.__curr_batch_size)]
        l_Leg = tf.gather_nd(last_output, indices, name=None) # left leg

        hierarchical_input = [chest_head, r_Arm, l_Arm, r_Leg, l_Leg]
        
        # Then pass through the network

        ################    First layer encoding    #################
        first_hidden_layer = []
        for bp_id in range(len(self.__encode1)): # bp_id - body part ID
          name_of_body_part = self._body_part_names[bp_id]
          first_hidden_layer.append(self._activate(hierarchical_input[bp_id], self["w_"+name_of_body_part], self["b_"+name_of_body_part]))

        # Combine the outputs
        spine_and_r_arm = tf.concat(1, [first_hidden_layer[0], first_hidden_layer[1]])
        spine_and_l_arm = tf.concat(1, [first_hidden_layer[0], first_hidden_layer[2]])
        spine_and_r_leg = tf.concat(1, [first_hidden_layer[0], first_hidden_layer[3]])
        spine_and_l_leg = tf.concat(1, [first_hidden_layer[0], first_hidden_layer[4]])
        hierarchical_hidden_layer = [spine_and_r_arm, spine_and_l_arm, spine_and_r_leg, spine_and_l_leg]

        ################    Second layer encoding    #################
        second_hidden_layer = []
        for ind in range(len(self.__encode2)): # ind - index
          name_of_part = "spine_"+self._body_part_names[ind+1] # skip head and chest
          #print("Encode " + name_of_part)
          second_hidden_layer.append(self._activate(hierarchical_hidden_layer[ind], self["w_"+name_of_part], self["b_"+name_of_part]))

        # Combine the outputs
        upper_body_in = tf.concat(1, [second_hidden_layer[0], second_hidden_layer[1]],name = 'upper_body_in')
        lower_body_in = tf.concat(1, [second_hidden_layer[2], second_hidden_layer[3]],name = 'lower_body_in')

        ################    Thirsd layer encoding    #################
        upper_body_out = self._activate(upper_body_in, self["w_upper_body"], self["b_upper_body"])
        lower_body_out = self._activate(lower_body_in, self["w_lower_body"], self["b_lower_body"])

        # Concatanate
        whole_body_in = tf.concat(1, [body_in, upper_body_out, lower_body_out],name = 'whole_body_in')

        ################    4th layer encoding    ##################
        representation = self._activate(whole_body_in, self["w_whole_body"], self["b_whole_body"])

        if(just_middle):
          return representation

        #########################            DECODING                ######################3

        ################    1st layer decoding    ##################
        whole_body_decode = self._activate(representation , self["w_whole_body_decode"], self["b_whole_body_decode"])

        # Slice it back into 3 parts : upper, lower body and the bits, which are affection the whole body
        """indices = [ [ [elem, index] for index in range(self._body_channels) ] for elem in range(self.__curr_batch_size)]
        body_decode = tf.gather_nd(whole_body_decode, indices, name=None) # changes the whole body
        indices = [ [ [elem, index] for index in range(self._body_channels, self._body_channels+FLAGS.upper_body_neurons,1) ] for elem in range(self.__curr_batch_size)]
        upper_body_decode = tf.gather_nd(whole_body_decode, indices, name=None) # changes the whole body
        indices = [ [ [elem, index] for index in range(self._body_channels+FLAGS.upper_body_neurons, 6+FLAGS.upper_body_neurons+FLAGS.lower_body_neurons,1) ] for elem in range(self.__curr_batch_size)]
        lower_body_decode = tf.gather_nd(whole_body_decode, indices, name=None) # changes the whole body"""

        body_decode = tf.slice(whole_body_decode, [0,0], [self.__curr_batch_size,self._body_channels], name = 'body_decode')
        upper_body_slice = tf.slice(whole_body_decode, [0,self._body_channels], [self.__curr_batch_size,FLAGS.upper_body_neurons], name = 'upper_body_slice')
        lower_body_slice = tf.slice(whole_body_decode, [0,self._body_channels+FLAGS.upper_body_neurons], [self.__curr_batch_size,FLAGS.lower_body_neurons], name = 'lower_body_slice')

        ################    2nd layer decoding    ##################
        upper_body_decode = self._activate(upper_body_slice, self["w_upper_body_decode"], self["b_upper_body_decode"])
        lower_body_decode = self._activate(lower_body_slice, self["w_lower_body_decode"], self["b_lower_body_decode"])

        """ # Slice it back
        spine_and_r_arm_slice = tf.slice(upper_body_decode, [0,0], [self.__curr_batch_size, self.__encode2[0]])
        spine_and_l_arm_slice = tf.slice(upper_body_decode, [0,self.__encode2[0]], [self.__curr_batch_size, self.__encode2[1]])
        spine_and_r_leg_slice = tf.slice(upper_body_decode, [0,self.__encode2[0] + self.__encode2[1]], [self.__curr_batch_size, self.__encode2[2]])
        spine_and_l_leg_slice = tf.slice(upper_body_decode, [0,self.__encode2[0] + self.__encode2[1]+ self.__encode2[2]], [self.__curr_batch_size, self.__encode2[3]])

        ################    3rd layer decoding    ##################
        spine_and_r_arm_decode = self._activate(spine_and_r_arm_slice, self["w_spine_r_arm_decode"], self["b_spine_r_arm_decode"])
        spine_and_l_arm_decode = self._activate(spine_and_l_arm_slice, self["w_spine_l_arm_decode"], self["b_spine_l_arm_decode"])
        spine_and_r_leg_decode = self._activate(spine_and_r_leg_slice, self["w_spine_r_leg_decode"], self["b_spine_r_leg_decode"])
        spine_and_l_leg_decode = self._activate(spine_and_l_leg_slice, self["w_spine_l_leg_decode"], self["b_spine_l_leg_decode"])

        # Average slices from all the layers, since spine was recombined in all of them
        spine_slice = 0,25 * (tf.slice(spine_and_r_arm_decode,[0,0],[self.__curr_batch_size, self.__encode1[0]]) +
                              tf.slice(spine_and_l_arm_decode,[0,0],[self.__curr_batch_size, self.__encode1[0]]) +
                              tf.slice(spine_and_r_leg_decode,[0,0],[self.__curr_batch_size, self.__encode1[0]]) +
                              tf.slice(spine_and_l_leg_decode,[0,0],[self.__curr_batch_size, self.__encode1[0]]) )"""

        # Slice it back to the body parts
        spine_slice = tf.slice(upper_body_decode,[0,0],[self.__curr_batch_size, self.__encode1[0]], name = 'spine_decode_slice')
        r_arm_slice = tf.slice(upper_body_decode,[0,self.__encode1[0]],[self.__curr_batch_size, self.__encode1[1]])
        l_arm_slice = tf.slice(upper_body_decode,[0,self.__encode1[0]+self.__encode1[1]],[self.__curr_batch_size, self.__encode1[2]])
        r_leg_slice = tf.slice(lower_body_decode,[0,0],[self.__curr_batch_size, self.__encode1[3]])
        l_leg_slice = tf.slice(lower_body_decode,[0,self.__encode1[3]],[self.__curr_batch_size, self.__encode1[4]])
        

        ################    4rd decoding  layer  ##################
        r_arm_decode = self._activate(r_arm_slice, self["w_r_arm_decode"], self["b_r_arm_decode"])
        l_arm_decode = self._activate(l_arm_slice, self["w_l_arm_decode"], self["b_l_arm_decode"])
        r_leg_decode = self._activate(r_leg_slice, self["w_r_leg_decode"], self["b_r_leg_decode"])
        l_leg_decode = self._activate(l_leg_slice, self["w_l_leg_decode"], self["b_l_leg_decode"])
        chest_head_decode = self._activate(spine_slice, self["w_chest_head_decode"], self["b_chest_head_decode"])

        output = tf.concat(1, [body_decode, chest_head_decode, r_arm_decode, l_arm_decode,r_leg_decode,l_leg_decode], name='concat')
      #print('I passed')

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
      self.__curr_batch_size = len(currSequence)
      print("Length of curr seq. : " + str(self.__curr_batch_size))

      # Define on an operator
      middle_op = self.run_net(input_ , 1, just_middle = True) # 1 means that we have no dropout
        
      # for each snippet in a sequence
      # pass through the network untill the middle layer
      middle = sess.run(middle_op, feed_dict={input_: currSequence})
        
      # save it into a file
      sio.savemat(output_seq_file_name, {'trialId':name, 'spikes':np.transpose(middle)})  
