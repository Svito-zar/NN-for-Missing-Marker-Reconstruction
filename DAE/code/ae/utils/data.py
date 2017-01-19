"""Functions for downloading,reading and preprocessing CMU data."""

from __future__ import division
from __future__ import print_function

from collections import deque

import gzip

import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from flags import FLAGS
import os

import sys # for adding a python module from the folder
sys.path.append('/home/taras/Desktop/Work/Code/Git/DAE-for-Mocap-Representation-Learning/BVH_format/parser') # address of the BVH parser
#sys.path.append('/home/taras/Code/Git/DAE-for-Mocap-Representation-Learning/BVH_format/parser') # address of the BVH parser
from reader import MyReader


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):

  def __init__(self, sequences, labels, fake_data=False):
    if fake_data:
      self._num_sequences = 10000
    else:
      assert sequences.shape[0] == labels.shape[0], (
          "sequences.shape: %s labels.shape: %s" % (sequences.shape,
                                                 labels.shape))
      self._num_sequences = sequences.shape[0]

    self._sequences = sequences
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._min = 0 # the smallest value in the dataset
    self._max = 0 # the biggest values in the dataset

  @property
  def sequences(self):
    return self._sequences

  @property
  def labels(self):
    return self._labels

  @property
  def num_sequences(self):
    return self._num_sequences

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_sequence(self):
    """Return the next sequence from this data set."""
    sequence_numb= self._index_in_epoch
    self._index_in_epoch += 1
    if self._index_in_epoch > self._num_sequences:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_sequences)
      np.random.shuffle(perm)
      self._sequences = self._sequences[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      sequence_numb= 0
      self._index_in_epoch = 1
    return self._sequences[sequence_numb], self._labels[sequence_numb]


class DataSetPreTraining(object):

  def __init__(self, sequences):
    self._sequences = sequences
    self._num_sequences = sequences.shape[0]
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def sequences(self):
    return self._sequences

  @property
  def num_sequences(self):
    return self._num_sequences

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_sequence(self):
    """Return the next sequence from this data set."""
    sequence_numb= self._index_in_epoch
    self._index_in_epoch += 1
    if self._index_in_epoch > self._num_sequences:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_sequences)
      np.random.shuffle(perm)
      self._sequences = self._sequences[perm]
      # Start next epoch
      sequence_numb = 0
      self._index_in_epoch = 1
    return self._sequences[sequence_numb]

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
    array = np.array(reader.points) # use 3d coordinates
    # Check amount of dimensions
    if(array.ndim != 3):
        raise ValueError('Input array from the bvhReader is not 3d !')
    # Reshape : T * D * N -> T * N * D
    timeSteps, dimensionality, jointsAmount = array.shape
    array = np.transpose(array,(0,2,1))
    # Reshape : T * N * D -> T * ND
    sequence = np.reshape(array, (timeSteps,dimensionality * jointsAmount))
    return sequence

def read_unlabeled_data(train_dir, amount_of_subfolders):
  class DataSets(object):
    pass
  data_sets = DataSets()

  # go over all folders with the data
  print('Reading BVH files from ', FLAGS.amount_of_subfolders, ' folders : ' )
  train_dir = FLAGS.data_dir
  
   # It is very tricky to read the data into a list of arrays of a different sizes
   # so we are using a special structure deque() for that
  input_data = deque()
  numb_of_folders=FLAGS.amount_of_subfolders
  if(numb_of_folders>4):
      numb_of_folders=numb_of_folders+1 # since we skip the 4th one
      
  # go over all subfolders with the data putting them all into one list
  for folder_numb in range(1,numb_of_folders+1,1):
    if(folder_numb==4):
      continue
    if(folder_numb<10):
      curr_dir = train_dir+'/0'+str(folder_numb)
    else:
      curr_dir = train_dir+'/'+str(folder_numb)
    print(curr_dir)
    for filename in os.listdir(curr_dir):
      input_data.append(mean_normalization(read_file(curr_dir+'/'+filename)) )
                 
  
  firstSequence = input_data[0]
      
  amount_of_frames, DoF = firstSequence.shape

  amount_of_strings = len(input_data)

  if(DoF != FLAGS.DoF):
    raise ValueError(
          'Invalid amount of Degrees Of Freedom (DoF) %d in the FLAGS file!' %
          (FLAGS.DoF))
  
  print(str(amount_of_strings ) + ' sequences (' + str(DoF) + ' DoF) was read')

  #print('Example of a string : with ',amount_of_frames,' frames : \n', firstSequence)

  # Scales all values in the input_data to be between 0 and 1 """
  minimums = [array.min() for array in input_data]
  maximums = [array.max() for array in input_data]
  data_sets.min_val = min(minimums)
  input_data -= data_sets.min_val
  eps=1e-8
  data_sets.max_val = max(maximums)
  input_data *= 1.0 / (data_sets.max_val  + eps)
  #print('Minumum and maximum values in the dataset are : ',data_sets.min_val,data_sets.max_val)

    
  TEST_SEQUENCES_NUMBER = FLAGS.test_sequences_numb

  # TODO: do I need a validation dataset?
  VALIDATION_SIZE = 1 # FLAGS.validation_size

  # Shuffle the data
  perm = np.arange(amount_of_strings)
  np.random.shuffle(perm)
  input_data =  input_data[perm]

  train_data = input_data[TEST_SEQUENCES_NUMBER:]

  test_data = input_data[:TEST_SEQUENCES_NUMBER]

  validation_data = train_data[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:]

  data_sets.train = DataSetPreTraining(train_data)
  data_sets.validation = DataSetPreTraining(validation_data)
  data_sets.test = DataSetPreTraining(test_data)

  print (str(test_data.shape[0]) + ' sequences will be used for testing')
  print (str(validation_data.shape[0]) + ' sequences will be used for validation')
  print (str(train_data.shape[0]) + ' sequences will be used for training')
  
  return data_sets

''' Add Gaussian random vectors with zero mean and given variance '''
def _add_noise(x, variance):
  x_cp = np.copy(x)
  x_cp = x_cp + np.random.normal(0, variance,(x_cp.shape[0],
                                  x_cp.shape[1]))
  return x_cp


def fill_feed_dict_ae(data_set, input_pl, target_pl, keep_prob, variance, dropout, add_noise=True):
    input_feed = data_set.next_sequence()
    # allow no noise during testing
    if(add_noise):
      input_feed = _add_noise(input_feed, variance)
    feed_dict = {
        input_pl: input_feed,
        target_pl: input_feed,
        keep_prob: dropout
    }
    return feed_dict


def fill_feed_dict(data_set, images_pl, labels_pl, noise=False):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_sequence(FLAGS.batch_size)
  if noise:
      images_feed = _add_noise(images_feed, FLAGS.drop_out_rate)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

''' Split the single tensor of a sequence into a list of frames '''
def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

''' Do mean normalization : substract mean pose of the trial '''
def mean_normalization(array):
    mean = array.mean(axis=0)
    return array - mean[np.newaxis,:]
