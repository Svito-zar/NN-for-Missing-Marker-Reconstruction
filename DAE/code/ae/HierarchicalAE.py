from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import tensorflow as tf
from utils.flags import FLAGS
from utils.data import  add_noise, loss_reconstruction
from AE import AutoEncoder, simulate_missing_markets, use_existing_markers
#from train import DataInfo

class HierarchicalAE(AutoEncoder):
  """Hierarchical deep autoencoder.

  Deep Autoencoder(AE) with Hierarchical encoding, used for full training cycle.
  The user specifies the structure of the neural net
  by specifying number of layers and the number of hidden
  units for each layer.

  This AE will break the input skeleton into 5 body parts,
  recover missing markers for each of them separately
  and then fuse them together in the output
  """

  # The following two array must have the same size and ordering, they will be use to make a proper connection for the input layer
  _body_part_names = ["l_arm", "r_arm",  "spine", "l_leg", "r_leg" ]
  _output_shapes = [ 7, 7, 9, 9, 9]  # Hardcoded for the CMU data in c3d format

  def __init__(self, layers_width, layers_depth, variance_coef, data_info, sess):
    """Autoencoder initializer

      :param
      layers_width: the width of each part of hierarchical AE
      :param
      layers_depth: the depth of the hierarchical AE (same for each body part)
      :param
      sess: the tf session
  """

    AutoEncoder.__init__(self, layers_depth , FLAGS.batch_size, FLAGS.chunk_length, sess, data_info)


    self.__shape = [layers_width for i in range(layers_depth)]
    self.__width = layers_width
    self.__layers_depth = layers_depth

    #self.__recurrent_layer = FLAGS.recurrent_layer let's try to have it fully recurrent instead

    self.__variables = {}

    self._setup_model(variance_coef, data_info)

  def _setup_model(self, variance_coef, data_info):
    with tf.variable_scope("hiererchical_autoencoder_variables"):

      print('Setting up the variables for Hierarchical Autoencoder...\n')

      # *****************         Initialize weights and biases**********

      debug = False

      # Declare separate LSTM for each body part
      for bp_id in range(len(self._body_part_names)): # bp_id - body part ID

          name_of_part = self._body_part_names[bp_id]

          with tf.variable_scope("LSTM_"+name_of_part):

              # Define LSTM cell
              name_cell = name_of_part + "_cell"
              self[name_cell] = self.multi_lstm_cell(self._output_shapes[bp_id]*3)


      if(debug):
          print('The network was created')


      ##############        DEFINE THE NETWORK     ###############################################

      # Declare a mask for simulating missing_values
      self._mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.chunk_length, FLAGS.frame_size],
                                  name='Mask_of_mis_markers')
      self._mask_generator = self.binary_random_matrix_generator(FLAGS.missing_rate)

      # 1 - Setup network for TRAINing
      self._input_ = add_noise(self._train_batch, variance_coef, data_info._data_sigma)
      self._target_ = self._train_batch  # Input noisy data and reconstruct the original one

      # Define output and loss for the training data
      self._output = self.construct_graph(self._input_, test=False)  # process batch of sequences. no dropout

      # Get the tensor size
      shape = self._output.get_shape()
      size = shape[0] * shape[1] * shape[2]

      # Normalize the L2 loss
      self._reconstruction_loss = loss_reconstruction(self._output, self._target_, self.max_val) # / tf.cast(size, tf.float32)  # (batch_size*chunk_length)
      tf.add_to_collection('losses', self._reconstruction_loss)  # allow for weight decay
      self._loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

      # 2 - Setup network for TESTing
      self._valid_input_ = self._valid_batch
      self._valid_target_ = self._valid_batch
      tf.get_variable_scope().reuse_variables()
      self._valid_output = self.construct_graph(self._valid_input_, test=False)
      # Define loss
      self._valid_loss = loss_reconstruction(self._valid_output, self._valid_target_, self.max_val) # / tf.cast(size, tf.float32)  # (batch_size*chunk_length)

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

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def lstm_cell(self,size):
          basic_cell = tf.contrib.rnn.BasicLSTMCell(
              size, forget_bias=1.0, state_is_tuple=True)
          # Apply dropout on the hidden layers
          if (size != self.__shape[-1]):
              hidden_cell = tf.contrib.rnn.DropoutWrapper(cell=basic_cell, output_keep_prob=FLAGS.dropout)
              return hidden_cell
          else:
              return basic_cell


  def multi_lstm_cell(self, output_sizes):

      lstm_sizes = [self.__width for layer in range(self.__layers_depth)]
      lstm_sizes = np.append(lstm_sizes, output_sizes)

      return tf.contrib.rnn.MultiRNNCell(
          [self.lstm_cell(sz) for sz in lstm_sizes], state_is_tuple=True)

  def construct_graph(self, sequence_pl, test=False):

      # Extract the part of the data, which belongs to each body part
      # This part is format specific. And HARDCODED!
      # The data for the structure of the MoCap dataset (in c3d format) is in the file C3D_fierarchy in the root folder of my Dropbox folder

      # Simulate missing markers
      if (FLAGS.missing_markers_are_random):
          network_input = simulate_missing_markets(sequence_pl, self._mask, FLAGS.defaul_value)
      #else:
      #    network_input = remove_right_hand(sequence_pl)

      hierarchical_train_input = parse_hierarchy(self, network_input)

      # Go over body parts and run corresponding LSTMs
      for bp_id in range(0, len(self._body_part_names)):  # bp_id - body part ID

          name_of_part = self._body_part_names[bp_id]

          #print(name_of_part)

          with tf.variable_scope("LSTM_" + name_of_part):
              name_cell = name_of_part + "_cell"

              # Run LSTM cells for both input options
              outputs, _ = tf.nn.dynamic_rnn(
                  cell=self[name_cell],
                  dtype=tf.float32,
                  inputs=hierarchical_train_input[bp_id])  # tf.cast(, tf.float32))

              if (bp_id == 0):
                  train_output = outputs
              else:
                  train_output = tf.concat([train_output, outputs], 2)  # use actual spine values

              #print(train_output.shape)

      # Incorporate information from the input
      if(test):
          train_output = use_existing_markers(train_output,network_input, self._mask, FLAGS.defaul_value)

      tf.get_variable_scope().reuse_variables()  # so that we can use the same LSTM both for training and testing

      return train_output

def parse_hierarchy(ae, input_seq_pl):
    indices = [[[[elem, seq_id, index] for index in range(0, 7 * 3, 1)] for seq_id in range(ae.sequence_length)] for
               elem in range(ae.batch_size)]
    l_Arm = tf.gather_nd(input_seq_pl, indices, name='Exctract_l_arm')  # left hand

    indices = [[[[elem, seq_id, index] for index in range(7 * 3, 14 * 3, 1)] for seq_id in range(ae.sequence_length)]
               for elem in range(ae.batch_size)]
    r_Arm = tf.gather_nd(input_seq_pl, indices, name='Exctract_r_arm')  # right arm

    indices = [[[[elem, seq_id, index] for index in range(14 * 3, 27 * 3, 1)] +
                [[elem,seq_id, index] for index in range(0, 1 * 3, 1)] +  # left shoulder
                [[elem, seq_id, index] for index in range(7*3, 8 * 3, 1)] # right shoulder
                for seq_id in range(ae.sequence_length)]
               for elem in range(ae.batch_size)]

    spine = tf.gather_nd(input_seq_pl, indices, name='Exctract_spine')  # spine

    indices = [[[[elem, seq_id, index] for index in range(27 * 3, 34 * 3, 1)] +
                [[elem, seq_id, index] for index in range(23 * 3, 25 * 3, 1)]  # left hips
               for seq_id in range(ae.sequence_length)]
               for elem in range(ae.batch_size)]
    l_Leg = tf.gather_nd(input_seq_pl, indices, name='Exctract_l_leg')  # right leg

    indices = [[[[elem, seq_id, index] for index in range(34 * 3, 41 * 3, 1)] +
                [[elem, seq_id, index] for index in range(25 * 3, 27 * 3, 1)]  # right hips
                    for seq_id in range(ae.sequence_length)]
                    for elem in range(ae.batch_size)]
    # print(np.array(indices).shape)
    r_Leg = tf.gather_nd(input_seq_pl, indices, name='Exctract_r_leg')  # left leg

    hierarchical_input = [l_Arm, r_Arm,  spine, l_Leg, r_Leg]

    return hierarchical_input

    #TODO: test dropout

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

if __name__ == '__main__':
    print('You have run Hierarchical AE as a main file')


    # Simulate the information about the dataset
    data_info = DataInfo(np.ones([32,129]), [3000,129], [3000,129])

    sess= tf.Session()

    ae = HierarchicalAE(FLAGS.network_width, FLAGS.num_hidden_layers, FLAGS.variance_of_noise, data_info, sess)
    print('\nHierarchical AE was created !')

    test = read_c3d_file('/home/taras/Documents/Datasets/CMU_c3d/Walk/train/1/01_01.c3d')

    print(test.shape)

    # REshape into batches of sequences
    # Split it into chunks
    all_chunks = np.array([test[i:i + ae.sequence_length, :] for i in
                           xrange(0, len(test) - ae.sequence_length + 1,
                                  FLAGS.chunking_stride)])  # Split sequence into chunks

    # Batch those chunks
    batches = np.array(
        [all_chunks[i:i + ae.batch_size, :] for i in xrange(0, len(all_chunks) - ae.batch_size + 1, ae.batch_size)])


    if (FLAGS.Layer_wise_Pretraining):
        print('\nERROR! We cannot do layerwise pretraining for the hierarchical AE! Please, change the flags')
        exit(1)

    print(batches[0].shape)

    [spine, l_Arm, r_Arm, l_Leg, r_Leg] = parse_hierarchy(ae, batches[0])

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(test[0][0:123:3], test[0][1:123:3], test[0][2:123:3], c='r', marker='o')

    '''with sess.as_default():
        ax.scatter(l_Arm[0][0][0:21:3].eval(), l_Arm[0][0][1:21:3].eval(), l_Arm[10][2][2:21:3].eval(),
               c='r', marker='o')
        ax.scatter(r_Arm[0][0][0:21:3].eval(), r_Arm[0][0][1:21:3].eval(), r_Arm[10][2][2:21:3].eval(),
                   c='r', marker='o')
        ax.scatter(spine[0][0][0:51:3].eval(), spine[0][0][1:51:3].eval(), spine[10][2][2:51:3].eval(),
                   c='b', marker='o')'''
    #ax.scatter(all_3d_coords[110][0][treshhold:43], all_3d_coords[10][1][treshhold:43],
     #          all_3d_coords[10][2][treshhold:43], c='b', marker='o')

    plt.show()