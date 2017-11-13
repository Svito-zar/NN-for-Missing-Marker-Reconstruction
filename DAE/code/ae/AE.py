from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils.flags import FLAGS
import random


class AutoEncoder(object):
  """Generic autoencoder.

  It is am empty class
  to be parent for the Flat and Hierarchical AE.
  """
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  def __init__(self, num_hidden_layers, batch_size, sequence_length,  sess, data_info):
    """Autoencoder initializer

    Args:
      num_hidden_layers:          number of hidden layers
      batch_size:     batch size
      sequence_length: length of the sequence which will be feeded into LSTM as once
      sess:           tensorflow session object to use
    """

    self.__num_hidden_layers = num_hidden_layers
    self.__batch_size = batch_size
    self.__sequence_length = sequence_length

    self.__variables = {}
    self.__sess = sess

    self.__scaling_factor = 0.1
    self.__default_value = FLAGS.defaul_value

    self.__max_val = data_info._max_val

    #################### Add the DATASETS to the GRAPH ###############3

    #### 1 - TRAIN ###
    self._train_data_initializer = tf.placeholder(dtype=tf.float32, shape=data_info._train_shape)  # 1033 at home
    self._train_data = tf.Variable(self._train_data_initializer, trainable=False, collections=[],
                                   name='Train_data')
    if (FLAGS.Layer_wise_Pretraining):  # Have more epochs: also for the pretraining
        train_frames = tf.train.slice_input_producer([self._train_data],
                                                     num_epochs=FLAGS.training_epochs + FLAGS.num_hidden_layers * FLAGS.pretraining_epochs)
    else:
        train_frames = tf.train.slice_input_producer([self._train_data], num_epochs=FLAGS.training_epochs)
    self._train_batch = tf.train.shuffle_batch(train_frames, batch_size=FLAGS.batch_size, capacity=5000,
                                         min_after_dequeue=1000, name='Train_batch')

    #### 2 - VALIDATE ###
    self._valid_data_initializer = tf.placeholder(dtype=tf.float32, shape=data_info._eval_shape)  # 1033 at home
    self._valid_data = tf.Variable(self._valid_data_initializer, trainable=False, collections=[],
                                   name='Valid_data')
    valid_frames = tf.train.slice_input_producer([self._valid_data], num_epochs=FLAGS.training_epochs)
    self._valid_batch = tf.train.shuffle_batch(valid_frames, batch_size=FLAGS.batch_size, capacity=5000,
                                         min_after_dequeue=1000, name='Valid_batch')

  def construct_graph(self, input_seq_pl, dropout, test=False, just_middle = False):

          """Get the output of the autoencoder

          This is a dummy function, which has to be defined in each specific class, which inherits class AE

          Args:
            input_seq_pl:     tf placeholder for ae input data of size [batch_size, sequence_length, DoF]
            dropout:          how much of the input neurons will be activated, value in [0,1]
            just_middle :     will indicate if we want to extract only the middle layer of the network
          Returns:
            input_seq_pl
          """

          return input_seq_pl

  def binary_random_matrix_generator(self, prob_of_missing, train_flag):
      """ Generate a binary matrix with random values: 0 with the probability to have a missing marker

        Args:
          prob_of_missing: probability to have a missing marker
          train_flag:  indicator if we are in the training or testing
        Returns:
          mask : binary matrix to be multiplied on input in order to simulate missing markers
      """

      tensor_size = [FLAGS.batch_size, FLAGS.chunk_length, FLAGS.frame_size * FLAGS.amount_of_frames_as_input]

      # use 10 to color only the spine, 16 - spine and right hand, 22 - spine and both arms, 27 - all except left leg, 32 - all
      r_arm = np.array([10, 11, 12, 13, 14, 15])
      l_arm = np.array([16, 17, 18, 19, 20, 21])
      r_leg = np.array([22, 23, 24, 25, 26])
      l_leg = np.array([27, 28, 29, 30, 31])

      if(train_flag):
          # Define all the body parts
          body_parts = [r_arm, l_arm, r_leg, l_leg]
      else:
          # Define just one body part
          arms = np.array([10, 11, 12, 13, 14, 21])
          body_parts = [r_arm]

      # initialize mask
      mask = np.ones(tensor_size)

      # Generate missing markers matrix for each sequence in the batch
      for sequence in range(FLAGS.batch_size):
          # Choose a random body part
          missing_body_part = random.choice(body_parts)

          # create binary raw
          prev_data = np.ones([FLAGS.chunk_length, missing_body_part[0]*3] ) # we ignore FLAGS.amount_of_frames_as_input for now
          missing_data = np.zeros([FLAGS.chunk_length, (missing_body_part[-1] + 1 - missing_body_part[0])*3 ])

          Left_part = False
          
          if(Left_part and not train_flag):
            # Choose a random body part
            missing_body_part = r_leg

            # create binary raw
            prev_data_2 = np.ones([FLAGS.chunk_length, missing_body_part[0]*3 - (r_arm[-1]+1)*3] ) # we ignore FLAGS.amount_of_frames_as_input for now
            missing_data_2 = np.zeros([FLAGS.chunk_length, (missing_body_part[-1] + 1 - missing_body_part[0])*3 ])

            next_data = np.ones([FLAGS.chunk_length, (FLAGS.frame_size -(1 + missing_body_part[-1]) *3)] )
            mask_for_seq = np.concatenate((prev_data, missing_data,prev_data_2, missing_data_2,next_data), axis = 1)
            
          else:
            next_data = np.ones([FLAGS.chunk_length, (FLAGS.frame_size -(1 + missing_body_part[-1]) *3)] )
            mask_for_seq = np.concatenate((prev_data, missing_data,next_data), axis = 1)

          if (FLAGS.amount_of_frames_as_input > 1):
              mask_for_seq = np.tile(mask_for_seq, (1,FLAGS.amount_of_frames_as_input))

          # Add this raw to the binary mask
          mask[sequence] = mask_for_seq

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
    y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b)) # was sigmoid before
    return y

def remove_right_hand(input_position):
  """ Remove the coordinates of the right hand from the sequence frame

  Args:
    input_position: full body position
  Returns:
    position_wo_r_hand : position, where right hand was removed (now - smaller dimensionality)
  """

  # Define just one body part
  missing_body_part = np.array([7, 8, 9, 10]) - 1 # right arm

  # Go over all the frames in the input
  for frame_id in range(FLAGS.amount_of_frames_as_input):
    offset = FLAGS.frame_size * frame_id
    coords_before_right_arm = input_position[:,:, 0 + offset : missing_body_part[0] * 3 + offset ]
    coords_after_right_arm = input_position[:,:, (missing_body_part[-1] + 1) + offset : 66 + offset]
    frame_wo_r_hand = tf.concat((coords_before_right_arm, coords_after_right_arm), axis=2)
    # add next frame with r hand nullified to the resulting position
    if(frame_id) == 0:
      position_wo_r_hand = frame_wo_r_hand
    else:
      position_wo_r_hand = tf.concat((position_wo_r_hand, frame_wo_r_hand), axis = 2)

  return position_wo_r_hand


def simulate_missing_markets(input_position, mask, const):

    output = tf.multiply(input_position, mask)

    if const == 0:
        return output
    else:
        high_values = tf.multiply(1-mask, const)
        output = tf.add(output, high_values, 'Simulate_missing_markers_as_' + str(const))
        return output

def use_existing_markers(input, result, mask, const):
    """ We can use the information we have in place of the markers we know instead of the output of the network

       Args:
         input: the data we have
         result: the output of the network
         mask:  the binary matrix of missing markers
       Returns:
         output : the new body position, which takes the input into account
     """

    # Calculate what we have actually got from the network

    result_without_markers_we_had = np.multiply(result, 1 - mask)
    the_marker_we_had = np.multiply(input, mask)

    if(const==0):
        output = the_marker_we_had + result_without_markers_we_had
    else:
        # We need first to sunstract constant value from the "input"
        original_input = input -  tf.multiply(input, 1 - mask)
        # Now we are ready to combine them
        output = original_input + result_without_markers_we_had

    return output

if __name__ == '__main__':

    random_size = [2,2,2] #[FLAGS.batch_size, FLAGS.chunk_length, int(FLAGS.frame_size / 3)]
    tensor_size = [2,2,6] # [FLAGS.batch_size, FLAGS.chunk_length, FLAGS.frame_size]

    prob_of_missing = 0.2

    AE = AutoEncoder(3,16,32,tf.get_default_session())
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
