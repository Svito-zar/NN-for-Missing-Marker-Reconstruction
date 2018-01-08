"""Functions for downloading,reading and preprocessing CMU data."""

from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import sys # for adding a python module from the folder
import btk
import numpy as np
from flags import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

sys.path.append('/home/taras/Dropbox/Taras/2017_PhD_at_KTH/Code/Git/AppliedProbabilisticMachineLearningRPL/DAE-for-Representation-Learning/BVH_format/parser')
from reader import MyReader

class DataSet(object):

  def __init__(self, sequences, batch_size):
    self._batch_size = batch_size
    self._sequences = sequences             # all the sequnces in the dataset
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

def read_bvh_file(fileName, test=False):
    """
       Reads a file from CMU MoCap dataset in BVH format

       Returns:
            sequence [sequence_length,frame_size] - local chanells transformed to the hips-centered coordinates
            hips [frame_size] - coordinates of the hips

    """

    # Read the data
    reader = MyReader(fileName);
    reader.read();
    sequence = np.array(reader.points)

    # Remove unnecessary coords (fingers positions, which were not always constant)
    sequence = np.concatenate([sequence[:,:,0:16], sequence[:,:,29:35], sequence[:,:,47:]], axis = 2)

    # Translate to the hips-center coordinate system
    hips = sequence[:,:,0]
    sequence = sequence - hips[:,:,np.newaxis]

    # This is a visualization for debug
    '''fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    treshhold = 22 # to show legs in a different color
    # use 10 to color only the spine, 16 - spine and right hand, 22 - spine and both arms, 27 - all except left leg, 32 - all
    time_step = 100
    ax.scatter(sequence[time_step ][2][0:treshhold],sequence[time_step ][0][0:treshhold], sequence[time_step ][1][0:treshhold],
               c='r', marker='o')
    ax.scatter(sequence[time_step ][2][treshhold:], sequence[time_step ][0][treshhold:], sequence[time_step ][1][treshhold:],
               c='g', marker='o')
    plt.show()'''

    # Transpose the last 2 dimensions
    sequence = np.transpose(sequence, axes = (0,2,1))

    #Flaten all the coords into one vector [T,3,m] -> [T,3m]
    return np.reshape(sequence,(sequence.shape[0],sequence.shape[1]*sequence.shape[2])),hips

def read_a_folder(train_dir):
    """
        Reads a folder from CMU MoCap dataset in BVH format

        Args:
            train_dir - address to the train, dev and eval datasets

        Returns:
            data [N, sequnce_length, frame_size] - an array,containing preprocessed dataset

    """
    chunk_length = FLAGS.chunk_length
    stride = FLAGS.chunking_stride

    data = np.array([])

    for sub_dir in os.listdir(train_dir):
        print(sub_dir)
        for filename in os.listdir(train_dir + '/' + sub_dir):
            curr_sequence,_ = read_bvh_file(train_dir + '/' + sub_dir + '/' + filename)

            curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in
                                    xrange(0, len(curr_sequence) - chunk_length, stride)])  # Split sequence into chunks by sliding window

            if (curr_chunks.shape[0] > 0):

                # Concatenate curr chunks to all of them
                data = np.vstack([data, curr_chunks]) if data.size else np.array(curr_chunks)

        print(data.shape)

        if(data.shape[0]>100000): # do not want to have more sequences
            break

    data = np.array(data)

    return data

def read_unlabeled_data(train_dir, evaluate):
  """
    Reads 3 datasets: "Train","Dev" and "Eval" from the CMU MoCap dataset in bvh format
    
    Args:
        train_dir - address to the train, dev and eval datasets
        evaluate - flag : weather we want to evaluate a network or we just optimize hyper-parameters
    Returns:
        datasets - object, containing Train, Dev and Eval datasets
        max_val - maximal value in the raw data ( for post-processing)
        mean_pose - mean pose in the raw data ( for post-processing)

  """
  class DataSets(object):
    pass
  data_sets = DataSets()

  # Get constants from the file
  data_dir = FLAGS.data_dir
  chunk_length =FLAGS.chunk_length
  stride = FLAGS.chunking_stride

  if(stride > chunk_length):
    print('ERROR! \nYou have stride bigger than lentgh of chunks. Please, change those values at flags.py, so that you don\'t ignore the data')
    exit(0)

  #         #########             Get TRAIN data                  ###########
  print('\nReading train data from the following folder ... ', data_dir )
  train_data = read_a_folder(data_dir+'/train')

  [amount_of_train_strings, seq_length, DoF] = train_data.shape
  print('\n' + str(amount_of_train_strings) + ' sequences with length ' + str(seq_length) + ' will be used for training')


  #         #########             Get TEST data                  ###########
  
  print('\nReading test data from the following folder : ' )
  if(evaluate):
    test_data = read_a_folder(data_dir+'/eval')
  else:
    test_data = read_a_folder(data_dir+'/dev')

  [amount_of_test_strings, seq_length, DoF] = test_data.shape
  print('\n' + str(amount_of_test_strings) + ' sequences with length ' + str(seq_length) + ' will be used for testing')

  # Do mean normalization : substract mean pose
  mean_pose = train_data.mean(axis=(0,1))
  train_data = train_data - mean_pose[np.newaxis,np.newaxis,:]
  test_data = test_data - mean_pose[np.newaxis,np.newaxis,:]

  # Scale all values in the input_data to be between -1 and 1
  eps=1e-8
  max_train = np.amax(np.absolute(train_data), axis=(0,1))
  max_test = np.amax(np.absolute(test_data), axis=(0,1))
  max_val = np.maximum(max_train, max_test)
  train_data =np.divide(train_data,max_val[np.newaxis,np.newaxis,:]+eps)
  test_data =np.divide(test_data,max_val[np.newaxis,np.newaxis,:]+eps)

  # Chech the data range
  max_ = test_data.max()
  min_ = test_data.min()

  print("MAximum value in the normalized dataset : " + str(max_))
  print("Minimum value in the normalized dataset : " + str(min_))

  data_sets.train = DataSet(train_data, FLAGS.batch_size)
  data_sets.test = DataSet(test_data, FLAGS.batch_size)

  # Assign variance
  data_sets.train.sigma = np.std(train_data, axis=(0,1))

  # Check if we have enough data
  if(data_sets.train._num_chunks < data_sets.train._batch_size):
      print('ERROR: We have got not enough data! Reduce batch_size or increase amount of subfolder you use.')
      exit(1)
  
  return data_sets, max_val, mean_pose

def add_noise(x, variance_multiplier, sigma):
  """
        Add Gaussian noise to the data

        Args:
            x - input vector
            variance_multiplier - coefficient to multiple on, when we calculate a variance of the noise
            sigma - variance of the dataset
        Returns:
            x - output vector, noisy data

  """
  eps=1e-15
  noise = tf.random_normal(x.shape, 0.0, stddev = np.multiply(sigma, variance_multiplier) + eps)
  x = x + noise
  return x

def write_binary(evaluate):
  """
        Reads 3 datasets: "Train","Dev" and "Eval" from the CMU MoCap dataset in bvh format
        And write them in the binary format.
        Will get the address of the folder with the data from flags.py

        Args:
            evaluate - flag, indicating weather we are going to evaluate the system or we are optimizing hyper-parameters
        Returns:
            will write binary files in the same folder as the original data

  """

  #Get the data
  data, max_val,mean_pose = read_unlabeled_data(FLAGS.data_dir,evaluate)

  # Write all important information into binary files

  # Datasets themselfs
  train_file = open(FLAGS.data_dir+'/train.binary', 'wb')
  data.train._sequences.tofile(train_file)
  train_file.close()

  eval_file = open(FLAGS.data_dir+'/eval.binary', 'wb')
  data.test._sequences.tofile(eval_file)
  eval_file.close()

  # Dataset properties
  
  sigma_file = open(FLAGS.data_dir+'/variance.binary', 'wb')
  data.train.sigma.tofile(sigma_file)
  sigma_file.close()

  max_val_file = open(FLAGS.data_dir+'/maximums.binary', 'wb')
  max_val.tofile(max_val_file)
  max_val_file.close()

  mean_file = open(FLAGS.data_dir+'/mean.binary', 'wb')
  mean_pose.tofile(mean_file)
  mean_file.close()
  
def read_binary_dataset(dataset_name):
  """
  Reads a dataset from the corresponding binary file

  """
  filename = FLAGS.data_dir+'/'+dataset_name+'.binary'

  # Read the dataset
  dataset= np.fromfile(filename)
  amount_of_frames = int(dataset.shape[0] /(FLAGS.chunk_length * FLAGS.frame_size * FLAGS.amount_of_frames_as_input))

  # Clip array so that it divides exactly into the inputs we want (frame_size * amount_of_frames_as_input)
  dataset = dataset[0:amount_of_frames * FLAGS.chunk_length * FLAGS.frame_size * FLAGS.amount_of_frames_as_input]

  # Reshape
  dataset = dataset.reshape(amount_of_frames, FLAGS.chunk_length, FLAGS.frame_size * FLAGS.amount_of_frames_as_input)
  return dataset

def read_all_the_data():
  """
    Reads all 3 datasets and their properties from binary file format
    
    Will take them from the corresponding file in the folder, which is defined by FLAGS.data_dir

  """
  class DataSets(object):
    pass
  data_sets = DataSets()

 
  #         #########             Get TRAIN data                  ###########
  
  train_data = read_binary_dataset('train')
  [amount_of_train_strings, seq_length, DoF] = train_data.shape
  print('\n' + str(amount_of_train_strings) + ' sequences with length ' + str(seq_length) + ' and ' + str(FLAGS.amount_of_frames_as_input) + ' frames in each will be used for training')


  #         #########             Get TEST data                  ###########
  
  test_data = read_binary_dataset('eval')
  [amount_of_test_strings, seq_length, DoF] = test_data.shape
  print(str(amount_of_test_strings) + ' sequences with length ' + str(seq_length) + ' will be used for testing')

  # Shuffle the data
  perm = np.arange(amount_of_train_strings)
  np.random.shuffle(perm)
  train_data =  train_data[perm]

  data_sets.train = DataSet(train_data, FLAGS.batch_size)
  data_sets.test = DataSet(test_data, FLAGS.batch_size)

  # Assign variance
  data_sets.train.sigma = np.std(train_data, axis=(0,1))

  # Read maximal value and mean pose before normalizatio
  max_val = np.fromfile( FLAGS.data_dir+'/maximums.binary')
  mean_pose = np.fromfile(FLAGS.data_dir+'/mean.binary')
  
  # Check if we have enough data
  if(data_sets.train._num_chunks < data_sets.train._batch_size):
      print('ERROR: We have got not enough data! Reduce batch_size or increase amount of subfolder you use.')
      exit(1)
  
  return data_sets, max_val, mean_pose

def write_test_seq_in_binary(input_file_name, output_file_name):
    # Datasets themselfs
    test_file = open(output_file_name, 'wb')
    test_seq, _ = read_bvh_file(input_file_name)
    test_seq.tofile(test_file)
    test_file.close()
    print("The test sequence was read from", input_file_name, " and written to", output_file_name)

def read_test_seq_from_binary(binary_file_name):
    # REad the sequence
    read_seq = np.fromfile(binary_file_name)
    # Reshape
    read_seq = read_seq.reshape(-1, FLAGS.frame_size)
    amount_of_frames = int(read_seq.shape[0] / (FLAGS.chunk_length * FLAGS.amount_of_frames_as_input))
    if(amount_of_frames > 0):
        # Clip array so that it divides exactly into the inputs we want (frame_size * amount_of_frames_as_input)
        read_seq = read_seq[0:amount_of_frames * FLAGS.chunk_length * FLAGS.amount_of_frames_as_input]

    # Reshape
    read_seq = read_seq.reshape(-1, FLAGS.frame_size * FLAGS.amount_of_frames_as_input)

    return read_seq

def loss_reconstruction(output, target, max_vals):
  """ Reconstruction error

  Args:
    output:    tensor of net output
    target:    tensor of net we are trying to reconstruct
    max_vals:  array of absolute maximal values in the dataset, is used for scaling an error to the original space
  Returns:
    Scalar tensor of mean eucledean distance
  """

  with tf.name_scope("reconstruction_loss"):
      net_output_tf = tf.convert_to_tensor(tf.cast(output, tf.float32), name='input')
      target_tf = tf.convert_to_tensor(tf.cast(target, tf.float32), name='target')
      # Euclidean distance between net_output_tf,target_tf
      error = tf.subtract(net_output_tf, target_tf)
      # Convert it back from the [-1,1] to original values
      error_scaled= tf.multiply(error, max_vals[np.newaxis, :] + 1e-15)
      squared_error = tf.reduce_mean(tf.square(error_scaled))
      return squared_error

if __name__ == '__main__':

    Test_bvh = False
    if(Test_bvh):

        result = read_bvh_file('/home/taras/Documents/Datasets/MoCap/BVH/Raw/train/32/32_02.bvh')

        new_file_name= '/home/taras/Documents/Data/cmuconvert-daz-01-09/01/no_hands.bvh'

        np.savetxt(new_file_name, result, fmt='%.5f', delimiter=' ')

        exit(0)

    Test =False

    if(Test):
        write_test_seq_in_binary('/home/taras/Documents/Datasets/MoCap/BVH/test_seq/102_03.bvh',
                                 FLAGS.data_dir + '/basketball.binary')
        write_test_seq_in_binary('/home/taras/Documents/Datasets/MoCap/BVH/test_seq/85_02.bvh',
                                 FLAGS.data_dir + '/salto.binary')
        '''write_test_seq_in_binary('/home/taras/Documents/Datasets/MoCap/Raw/Test_seq/14_01.c3d',
         FLAGS.data_dir + '/boxing.binary')'''

    else:
        write_binary(True)