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


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('\nExtracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


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

    return self._poses[start:end], self._poses[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)

  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)

  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)

  return data_sets

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
  print('Reading BVH files:')
  train_dir = FLAGS.data_dir
  input_data = np.array([])
  # go over all subfolders with the data
  for folder_numb in range(1,FLAGS.amount_of_subfolders+1,1):
    curr_dir = train_dir+'/0'+str(folder_numb)
    print('from the folder ' + curr_dir+'...')
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
  mean = input_data.mean(axis=0)
  input_data = input_data - mean[np.newaxis,:]

  # Scales all values in the input_data to be between 0 and 1 """
  input_data = input_data.copy()
  input_data -= input_data.min()
  eps=1e-8
  input_data *= 1.0 / (input_data.max() + eps)

  #print(input_data[0])
    
  TEST_SIZE = 3000
  VALIDATION_SIZE = 5000

  train_data = input_data[TEST_SIZE:,:]

  test_data = input_data[:TEST_SIZE,:]

  validation_data = train_data[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:]

  data_sets.train = DataSetPreTraining(train_data)
  data_sets.validation = DataSetPreTraining(validation_data)
  data_sets.test = DataSetPreTraining(test_data)

  print (str(test_data.shape[0]) + ' poses will be used for testing')
  print (str(validation_data.shape[0]) + ' poses will be used for validation')
  print (str(train_data.shape[0]) + ' poses will be used for training')
  
  return data_sets


def _add_noise(x, rate):
  x_cp = np.copy(x)
  pix_to_drop = np.random.rand(x_cp.shape[0],
                                  x_cp.shape[1]) < rate
  x_cp[pix_to_drop] = FLAGS.zero_bound
  return x_cp


def fill_feed_dict_ae(data_set, input_pl, target_pl, noise=None):
    input_feed, target_feed = data_set.next_batch(FLAGS.batch_size)
    if noise:
      input_feed = _add_noise(input_feed, noise)
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed
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
