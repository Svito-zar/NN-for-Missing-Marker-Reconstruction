"""Functions for downloading,reading and preprocessing CMU data."""

from __future__ import division
from __future__ import print_function

#from collections import deque

import gzip

import numpy as np
import tensorflow as tf

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from flags import FLAGS
import os

import sys # for adding a python module from the folder
sys.path.append('/home/taras/Desktop/Work/Code/Git/DAE-for-Mocap-Representation-Learning/BVH_format/parser') # address of the BVH parser
#sys.path.append('/home/taras/Code/Git/DAE-for-Mocap-Representation-Learning/BVH_format/parser') # address of the BVH parser
from reader import MyReader


class DataSetPreTraining(object):

  def __init__(self, sequences, batch_size):
    self._batch_size = batch_size
    self._sequences = sequences
    self._num_chunks = sequences.shape[0]
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def sequences(self):
    return self._sequences

  @property
  def num_chunks(self):
    return self._num_num_chunks

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self):
    """Return the next batch of sequences from this data set."""
    batch_numb = self._index_in_epoch
    self._index_in_epoch += self._batch_size
    #print('Starting sequence : ' + str(self._index_in_epoch) + ' out of ' + str( self._num_chunks))
    if self._index_in_epoch > self._num_chunks:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_chunks)
      np.random.shuffle(perm)
      self._sequences = self._sequences[perm]
      # Start next epoch
      batch_numb = 0
      self._index_in_epoch = self._batch_size
    return self._sequences[batch_numb:batch_numb+self._batch_size:1, :]

def read_file(fileName):
      
    """
    Reads a file from CMU MoCap dataset in BVH format
    
    Args:
        rawDataPoints - array T * D * N
        (3d coordinates of all joints for all time snippets)
    Returns:
        sequence - array T * ND

    """
    # Read the data
    reader = MyReader(fileName);
    reader.read();
    sequence = np.array(reader.channels) # use 3d coordinates

    # Ignore global rotation TODO: this has to be changed for the new data format
    shorter_sequence  = np.concatenate((sequence[:,0:3], sequence[:,6:]), axis=1)
      
    return shorter_sequence

def read_unlabeled_data(train_dir, amount_of_subfolders, evaluate):
  class DataSets(object):
    pass
  data_sets = DataSets()

  # Get constants from the file
  train_dir = FLAGS.data_dir
  numb_of_folders=FLAGS.amount_of_subfolders
  chunk_length =FLAGS.chunk_length
  stride = FLAGS.chunking_stride

  if(stride > chunk_length):
    print('ERROR! \nYou have stride bigger than lentgh of chunks. Please, change those values at flags.py, so that you don\'t ignore the data')
    exit(0)

  #         #########             Get TRAIN data                  ###########
  
  # go over all folders with the data, exept for the last one
  print('\nReading train data : BVH files from ', FLAGS.amount_of_subfolders, ' folders : ' )

  if(numb_of_folders>=4):
      numb_of_folders=numb_of_folders+1 # since we skip the 4th one

  train_data = np.array([])
  # go over all subfolders with the data putting them all into one list
  for folder_numb in range(1,numb_of_folders+1,1):
    if(folder_numb==4):
      continue
    if(folder_numb<10):
      curr_dir = train_dir+'/0'+str(folder_numb)
    else:
      curr_dir = train_dir+'/'+str(folder_numb)
    print(curr_dir)
    for filename in os.listdir(curr_dir ):
      curr_sequence = read_file(curr_dir+'/'+filename)
      curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in xrange(0, len(curr_sequence)-chunk_length, stride)]) # Split sequence into chunks
      # Concatanate curr chunks to all of them
      train_data = np.vstack([train_data, curr_chunks]) if train_data.size else np.array(curr_chunks)
      
  [amount_of_strings, seq_length, DoF] = train_data.shape

  if(DoF != FLAGS.DoF):
    raise ValueError(
          'Invalid amount of Degrees Of Freedom (%d) %d in the FLAGS file!' %
          (DoF, FLAGS.DoF))
  
  print('\n' + str(amount_of_strings) + ' sequences with length ' + str(seq_length) + ' (with ' + str(DoF) + ' DoF) was read')

  train_data = np.array(train_data)

  #         #########             Get TEST data                  ###########
  
  print('\nReading test data from the following folder : ' )
  test_data = np.array([])
  if(evaluate):
    curr_dir = train_dir+'/eval'
  else:
    curr_dir = train_dir+'/dev'
  print(curr_dir)
  for filename in os.listdir(curr_dir):
    curr_sequence = read_file(curr_dir+'/'+filename)
    curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in xrange(0, len(curr_sequence)-chunk_length, stride)]) # Split sequence into chunks
    # Concatanate curr chunks to all of them
    test_data = np.vstack([test_data, curr_chunks]) if test_data.size else np.array(curr_chunks)

  [amount_of_test_strings, seq_length, DoF] = test_data.shape
  
  print('\n' + str(amount_of_test_strings) + ' sequences with length ' + str(seq_length) + ' will be used for testing')

  test_data = np.array(test_data)

  #         #########             Get Vailidation data                  ###########

  if(evaluate):
    print('\nReading validation data from the following folder : ' )
    valid_data = np.array([])
    curr_dir = train_dir+'/dev'
    print(curr_dir)
    for filename in os.listdir(curr_dir):
      curr_sequence = read_file(curr_dir+'/'+filename)
      curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in xrange(0, len(curr_sequence)-chunk_length, stride)]) # Split sequence into chunks
      # Concatanate curr chunks to all of them
      valid_data = np.vstack([valid_data, curr_chunks]) if valid_data.size else np.array(curr_chunks)

    [amount_of_test_strings, seq_length, DoF] = valid_data.shape
    
    print('\n' + str(amount_of_test_strings) + ' sequences with length ' + str(seq_length) + ' will be used for validation (early stopping)')

    valid_data = np.array(valid_data)

  else:
    valid_data = test_data

  # Do mean normalization : substract mean pose
  print('\nNormalizing the data ...')
  mean_pose = train_data.mean(axis=(0,1))
  train_data = train_data - mean_pose[np.newaxis,np.newaxis,:]
  test_data = test_data - mean_pose[np.newaxis,np.newaxis,:]
  valid_data = valid_data - mean_pose[np.newaxis,np.newaxis,:]

  # Scales all values in the input_data to be between -1 and 1
  eps=1e-15
  max_train = np.amax(np.absolute(train_data), axis=(0,1))
  max_test = np.amax(np.absolute(test_data), axis=(0,1))
  max_val = np.maximum(max_train, max_test)
  train_data =np.divide(train_data,max_val[np.newaxis,np.newaxis,:]+eps)
  test_data =np.divide(test_data,max_val[np.newaxis,np.newaxis,:]+eps)
  valid_data = np.divide(valid_data,max_val[np.newaxis,np.newaxis,:]+eps)

  # Chech the data range
  max_ = test_data.max()
  min_ = test_data.min()

  #DEBUG
  print("MAximum value in the normalized test dataset : " + str(max_))
  print("Minimum value in the normalized test dataset : " + str(min_))

  #VALIDATION_SIZE = FLAGS.validation_sequences

  # Shuffle the data
  perm = np.arange(amount_of_strings)
  np.random.shuffle(perm)
  train_data =  train_data[perm]

  #validation_data = test_data # TODO: maybe change

  data_sets.train = DataSetPreTraining(train_data, FLAGS.batch_size)
  data_sets.validation = DataSetPreTraining(valid_data, FLAGS.batch_size)
  data_sets.test = DataSetPreTraining(test_data, FLAGS.batch_size)

  # Assign variance
  data_sets.train.sigma = np.std(train_data, axis=(0,1))

  # Check if we have enough data
  if(data_sets.train._num_chunks < data_sets.train._batch_size):
      print('ERROR: We have got not enough data! Reduce batch_size or increase amount of subfolder you use.')
      exit(1)
  
  return data_sets, max_val, mean_pose

''' Add Gaussian random vectors with zero mean and given variance '''
def _add_noise(x, variance_multiplier, sigma):
  x_cp = np.copy(x)
  eps=1e-15
  noise = np.random.normal(0, np.multiply(sigma, variance_multiplier) + eps,(x_cp.shape[0],
                                  x_cp.shape[1], x_cp.shape[2]))
  x_cp = x_cp + noise
  return x_cp


def fill_feed_dict_ae(data_set, input_pl, target_pl, keep_prob, variance, dropout, add_noise=True):
    input_feed = data_set.next_batch()
    # allow no noise during testing
    if(add_noise):
      input_feed = _add_noise(input_feed, variance, data_set.sigma) # pass both a coefficient and a sigma of the data
    feed_dict = {
        input_pl: input_feed,
        target_pl: input_feed,
        keep_prob: dropout
    }
    return feed_dict

def loss_reconstruction(output, target):
  """ Reconstruction error

  Args:
    output: tensor of net output
    target: tensor of net we are trying to reconstruct
  Returns:
    Scalar tensor of mean eucledean distance
  """
  with tf.name_scope("reconstruction_loss"):
      net_output_tf = tf.convert_to_tensor(output, name='input')
      target_tf = tf.convert_to_tensor(target, name='target')
      # Euclidean distance between net_output_tf,target_tf
      l2diff =  tf.nn.l2_loss(tf.subtract(net_output_tf, target_tf))
      return l2diff
