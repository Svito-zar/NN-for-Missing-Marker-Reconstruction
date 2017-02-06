"""Functions for downloading and reading CMU data."""

from __future__ import division
from __future__ import print_function

import gzip

import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from flags import FLAGS
import os

import sys # for adding a python module from the folder
sys.path.append('/home/taras/Documents/Code/BVH/parser') # address of the BVH parser
from reader import MyReader


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):

  def __init__(self, poses, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert poses.shape[0] == labels.shape[0], (
          "poses.shape: %s labels.shape: %s" % (poses.shape,
                                                 labels.shape))
      self._num_examples = poses.shape[0]

    self._poses = poses
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._min = 0 # the smallest value in the dataset
    self._max = 0 # the biggest values in the dataset

  @property
  def poses(self):
    return self._poses

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._poses = self._poses[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._poses[start:end], self._labels[start:end]


class DataSetPreTraining(object):

  def __init__(self, poses):
    self._num_examples = poses.shape[0]
    self._poses = poses
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def poses(self):
    return self._poses

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._poses = self._poses[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._poses[start:end]


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
    return sequence

def read_unlabeled_data(train_dir, amount_of_subfolders):
  """
    Reads a all the data from CMU MoCap dataset in BVH format
     
    Args:
        train_dir:              directore with the data
        amount_of_subfolders:   amount of sibfolders in this directory we want to use
    Returns:
        datasets:               object, containing data_sets.train, data_sets.test and data_sets.validation datasets
        min_val:                minimal value in the dataset
        max_val:                maximal value in the dataset
 
    """
  
  class DataSets(object):
    pass
  data_sets = DataSets()

  # go over all folders with the data
  print('Reading BVH files from ', FLAGS.amount_of_subfolders, ' folders : ' )
  train_dir = FLAGS.data_dir
  input_data = np.array([])
  numb_of_folders=FLAGS.amount_of_subfolders

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
      currSequence = read_file(curr_dir+'/'+filename)
      if input_data.size==0:
        input_data = np.array(currSequence)
      else:
        input_data =  np.concatenate((input_data,currSequence),0) 
  amount_of_frames, DoF = input_data.shape

  if(DoF != FLAGS.DoF):
    raise ValueError(
          'Invalid amount of Degrees Of Freedom (DoF) %d in the FLAGS file!' %
          (FLAGS.DoF))
  print(str(amount_of_frames) + ' frames (poses) with ' + str(DoF) + ' DoF was read')

  # Round amount of observations to the batchsize
  new_size = amount_of_frames - (amount_of_frames%FLAGS.batch_size)
  input_data = input_data[:new_size]

  # Do mean normalization : substract mean pose
  print('Normalizing the data ...')
  mean_pose = input_data.mean(axis=0)
  input_data = input_data - mean_pose [np.newaxis,:]

  # Scales all values in the input_data to be between -1 and 1
  max_val = input_data.max()
  eps=1e-8
  input_data *= 1.0 / (max_val + eps)

  # Get a standart deviation
  sigma = np.std(input_data, axis=0)

  # TODO: It should not be hardcoded
  TEST_SIZE = 5000
  VALIDATION_SIZE = 1000

  # Shuffle the data
  perm = np.arange(new_size)
  np.random.shuffle(perm)
  input_data =  input_data[perm]

  train_data = input_data[TEST_SIZE:,:]

  test_data = input_data[:TEST_SIZE,:]

  validation_data = train_data[:VALIDATION_SIZE]

  data_sets.train = DataSetPreTraining(train_data)
  data_sets.validation = DataSetPreTraining(validation_data)
  data_sets.test = DataSetPreTraining(test_data)

  # Assign variance
  data_sets.train.sigma = sigma

  #print (str(test_data.shape[0]) + ' poses will be used for testing')
  #print (str(validation_data.shape[0]) + ' poses will be used for validation')
  #print (str(train_data.shape[0]) + ' poses will be used for training')
  
  return data_sets, max_val, mean_pose



''' Add Gaussian random vectors with zero mean and given variance '''
def _add_noise(x, variance_multiplier, sigma):
  x_cp = np.copy(x)
  #print(x.shape)
  #print(np.multiply(sigma, variance_multiplier))
  eps=1e-15
  x_cp = x_cp + np.random.normal(0, np.multiply(sigma, variance_multiplier) + eps,(x_cp.shape[0],
                                  x_cp.shape[1]))
  return x_cp


def fill_feed_dict_ae(data_set, input_pl, target_pl, keep_prob, variance, dropout, add_noise=True):
    input_feed = data_set.next_batch(FLAGS.batch_size)
    # allow no noise during testing
    if(add_noise):
      input_feed = _add_noise(input_feed, variance, data_set.sigma) # pass both a coefficient and a sigma of the data
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
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  if noise:
      images_feed = _add_noise(images_feed, FLAGS.drop_out_rate)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict
