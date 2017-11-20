"""Functions for downloading,reading and preprocessing CMU data."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from flags import FLAGS
import os

import sys # for adding a python module from the folder
#import btk
import numpy as np
from flags import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

sys.path.append('/home/taras/Dropbox/Taras/2017_PhD_at_KTH/Code/Git/AppliedProbabilisticMachineLearningRPL/DAE-for-Representation-Learning/BVH_format/parser')
#sys.path.append('/home/taras/Documents/Code/Git/AppliedProbabilisticMachineLearningRPL/DAE-for-Representation-Learning/BVH_format/parser')
from reader import MyReader

class DataSet(object):

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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_bvh_file(fileName, test=False):
    """
       Reads a file from CMU MoCap dataset in BVH format

        Returns:
            shorter_sequence - local chanells of all the joins, except for the fingers
            global_rotation - chanells for the global rotaion

    """

    # Read the data
    reader = MyReader(fileName);
    reader.read();
    sequence = np.array(reader.points)

    # Remove unnecessary coords
    sequence = np.concatenate([sequence[:,:,0:16], sequence[:,:,29:35], sequence[:,:,47:]], axis = 2)

    # Translate to the hips-center coords\
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
    #Flaten all the coords into one vector
    return np.reshape(sequence,(sequence.shape[0],sequence.shape[1]*sequence.shape[2])), hips


def read_c3d_file(file_name):
    """
    Reads a file from CMU MoCap dataset in the c3d format

    Args:
        filename - adress of the file with the MoCap data in the c3d format
    Returns:
        sequence - array T * ND

    """

    # First read a set of marker labels, in order to filter out unnasesery one
    labels_file = open("MainLabels.txt", "r")

    debug = False

    # Read the data
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(file_name)  # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput()  # acq is the btk aquisition object

    #print('Reading a c3d file', file_name)

    all_3d_coords = np.array([])
    point_id = 0

    # Get subject name
    name = ''
    for i in range(0, acq.GetPoints().GetItemNumber()):
        label_string = acq.GetPoint(i).GetLabel()
        name_parts = len(label_string.split(':', 2))
        if(debug):
            print(label_string.split(':', 2))
        if(name_parts==2 and label_string.split(':', 2)[1] == "LSHO"):
            name = label_string.split(':', 2)[0] + ':'
            #break

    if (debug):
        print(name)

    # Check scaling factor
    file_instance = btk.btkC3DFileIO()
    file_instance.Read(file_name, acq)

    if (debug):
        print("Units:", acq.GetPointUnit())
    #print("Scale: ", file_instance.GetPointScale())

    missing_markers = list()

    while (True):
        try:
            # Get the next label
            label = name + labels_file.readline().splitlines()[0]
            if (debug and point_id < 7):
                print(label)
            next_point = acq.GetPoint(label).GetValues()
        except IndexError:
            print('Read', point_id, 'skeleton 3d points during', acq.GetPointFrameNumber(), 'time frames')
            break


        # Check if the data is there
        if(next_point[0][0] == 0 and next_point[0][1] == 0 and next_point[0][2] == 0):
            print('Problematic label was ', label)
            print('Last: ', next_point[-1])
            for time_step in range(next_point.shape[0]):
                if(next_point[time_step][0]==0):
                    missing_markers.append(point_id)
                    #next_point[time_step] = next_point[-1] # assign all the "zero" time-steps to the last one

        # Concatanate curr chunks to all of them
        all_3d_coords = np.dstack([all_3d_coords, next_point]) if all_3d_coords.size else np.array(next_point)

        point_id += 1

    print(missing_markers)

    # Convert to the hips centered coordinates
    hips = np.zeros([all_3d_coords.shape[0],3]) # coordinates of the hips at each time-step

    for time_step in range(all_3d_coords.shape[0]):
        # Obtain hips coords
        hips_x = np.average(all_3d_coords[time_step][0][23:27])
        hips_y = np.average(all_3d_coords[time_step][1][23:27])
        hips_z = np.average(all_3d_coords[time_step][2][23:27])
        hips[time_step] = [hips_x, hips_y, hips_z]
        #if(time_step==100):
        #    ax.scatter(hips_x, hips_y,hips_z, c='g', marker='x')
        # Subtract hips coors from each joint
        all_3d_coords[time_step]-=hips[time_step][:,np.newaxis]

    amount_of_frames = acq.GetPointFrameNumber()

    # Replace missing markers with the same marker at the end of the sequence
    for marker_id in missing_markers:
        for coord in range(3):
            if(marker_id>22 and marker_id<27): # if that are the hips
                all_3d_coords[time_step][coord][marker_id] = 0 # center - hips coords
            else:
                all_3d_coords[time_step][coord][marker_id] = all_3d_coords[amount_of_frames-1][coord][marker_id]


    # For debug - Visualize the skeleton
    '''fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    treshhold= 25
    ax.scatter(all_3d_coords[10][0][1:treshhold], all_3d_coords[10][1][1:treshhold], all_3d_coords[10][2][1:treshhold], c='r', marker='o')
    ax.scatter(all_3d_coords[10][0][treshhold:43], all_3d_coords[10][1][treshhold:43], all_3d_coords[10][2][treshhold:43], c='b', marker='o')
    plt.show()'''

    # Make a proper array shape
    mocap_seq = all_3d_coords.reshape(all_3d_coords.shape[0], -1) # Concatanate all coords into one vector
    '''print(mocap_seq.shape)
    amount_of_inputs = int(mocap_seq.shape[0] /  (FLAGS.frame_size * FLAGS.amount_of_frames_as_input))
    print(amount_of_inputs *  FLAGS.frame_size  * FLAGS.amount_of_frames_as_input)
    mocap_seq = mocap_seq[0:amount_of_inputs *  FLAGS.frame_size * FLAGS.amount_of_frames_as_input,:] # cut of what will not fit
    print(mocap_seq.shape)
    mocap_seq = mocap_seq.reshape(-1, FLAGS.frame_size * FLAGS.amount_of_frames_as_input) # join a few frames, if we take a couple of them at a time
    '''


    # Calculate max_val
    max = np.amax(np.absolute(mocap_seq))
    if(max > 100000):
        print("\nWATCH! The file ", file_name, " had maximal value ", max, "\n")

    return mocap_seq

def read_unlabeled_data(train_dir, evaluate):
  """
    Reads all 3 datasets from CMU MoCap dataset in BVH format
    
    Args:
        train_dir - address to the train, dev and eval datasets
        evaluate - flag : weather we want to evaluate a network or we just optimize hyperparameters
    Returns:
        datasets - object, containing Train, Dev and Eval datasets
        max_val - maximal value in the raw data ( for preprocessing)
        mean_pose - mean pose in the raw data ( for preprocessing)

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
  #print('\nNormalizing the data ...')
  mean_pose = train_data.mean(axis=(0,1))
  #print(mean_pose.shape)
  train_data = train_data - mean_pose[np.newaxis,np.newaxis,:]
  test_data = test_data - mean_pose[np.newaxis,np.newaxis,:]

  # Scales all values in the input_data to be between -1 and 1
  eps=1e-8
  max_train = np.amax(np.absolute(train_data), axis=(0,1))
  max_test = np.amax(np.absolute(test_data), axis=(0,1))
  max_val = np.maximum(max_train, max_test)
  train_data =np.divide(train_data,max_val[np.newaxis,np.newaxis,:]+eps)
  test_data =np.divide(test_data,max_val[np.newaxis,np.newaxis,:]+eps)

  #DEBUG
  # Chech the data range
  max_ = test_data.max()
  min_ = test_data.min()

  print("MAximum value in the normalized test dataset : " + str(max_))
  print("Minimum value in the normalized test dataset : " + str(min_))


  # Shuffle the data
  '''perm = np.arange(amount_of_train_strings)
  np.random.shuffle(perm)
  train_data =  train_data[perm]'''

  print('\nTrain data shape: ', train_data.shape)

  data_sets.train = DataSet(train_data, FLAGS.batch_size)
  data_sets.test = DataSet(test_data, FLAGS.batch_size)

  # Assign variance
  data_sets.train.sigma = np.std(train_data, axis=(0,1))

  # Check if we have enough data
  if(data_sets.train._num_chunks < data_sets.train._batch_size):
      print('ERROR: We have got not enough data! Reduce batch_size or increase amount of subfolder you use.')
      exit(1)
  
  return data_sets, max_val, mean_pose

def read_a_folder(curr_dir):
    chunk_length = FLAGS.chunk_length
    stride = FLAGS.chunking_stride

    data = np.array([])

    for sub_dir in os.listdir(curr_dir):
        print(sub_dir)
        for filename in os.listdir(curr_dir + '/' + sub_dir):
            curr_sequence,_ = read_bvh_file(curr_dir + '/' + sub_dir + '/' + filename)

            ''''# Move to TfRecords
            print(curr_sequence)
            print(curr_sequence.shape)

            write_Tf_records('train.tfrecords',curr_sequence)

            read_Tf_records('train.tfrecords')

            exit()'''

            curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in
                                    xrange(0, len(curr_sequence) - chunk_length, stride)])  # Split sequence into chunks
            if (curr_chunks.shape[0] > 0):
                # Concatanate curr chunks to all of them
                data = np.vstack([data, curr_chunks]) if data.size else np.array(curr_chunks)

        print(data.shape)

        if(data.shape[0]>10000):
            break

    data = np.array(data)

    return data

''' Add Gaussian random vectors with zero mean and given variance '''
def add_noise(x, variance_multiplier, sigma):
  eps=1e-15
  noise = tf.random_normal(x.shape, 0.0, stddev = np.multiply(sigma, variance_multiplier) + eps)
  x = x + noise
  return x


def remove_fingers(input_position):
  """ Extract all the DoF, but the fingers

  Args:
    input_position: full body position
  Returns:
    position_wo_fingers : position without fingers
  """

  left = input_position[:, 0:30]
  right = input_position[:,60:72]
  legs = input_position[:,102:126]
  position_wo_fingers = np.concatenate((left, right, legs), axis=1)
  return position_wo_fingers

def get_fingers(input_position):
  """ Extract the DoF, which correspond the fingers

  Args:
    input_position: full body position
  Returns:
    fingers : position of fingers
  """

  left_h = input_position[:,:, 30:60]
  right_h = input_position[:,:,72:102]
  fingers = np.concatenate((left_h, right_h), axis=2)
  return fingers

def write_binary():
  #Get the data
  data, max_val,mean_pose = read_unlabeled_data(FLAGS.data_dir,False) #read_all_the_data()

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
  #print('sigma:',  data.train.sigma[0:20])

  max_val_file = open(FLAGS.data_dir+'/maximums.binary', 'wb')
  max_val.tofile(max_val_file)
  max_val_file.close()
  #print('Max:',  max_val[0:20])

  mean_file = open(FLAGS.data_dir+'/mean.binary', 'wb')
  mean_pose.tofile(mean_file)
  mean_file.close()
  #print('Mean:',  mean_pose[0:20])


def read_sigma():
  sigma= np.fromfile(FLAGS.data_dir+'/variance.binary')
  print(sigma[0:20])
  
def read_binary_dataset(dataset_name):
  filename = FLAGS.data_dir+'/'+dataset_name+'.binary'
  #print('\nReading ', dataset_name, 'dataset from the following file... ' , filename)
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
    
    Will take them from the corresponding file in the folder set by FLAGS.data_dir

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
    #test_seq = read_c3d_file(input_file_name)'''
    test_seq, _ = read_bvh_file(input_file_name)
    test_seq.tofile(test_file)
    test_file.close()
    print("The test sequence was read from", input_file_name, " and written to", output_file_name)

def read_test_seq_from_binary(binary_file_name):
    # REad the sequence
    read_seq = np.fromfile(binary_file_name)
    # Reshape
    read_seq = read_seq.reshape(-1, FLAGS.frame_size)
    #print("The test sequence was read from", binary_file_name)

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
    output: tensor of net output
    target: tensor of net we are trying to reconstruct
  Returns:
    Scalar tensor of mean eucledean distance
  """

  if (FLAGS.amount_of_frames_as_input > 1):
      max_vals = np.tile(max_vals, FLAGS.amount_of_frames_as_input)

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
    #write_binary()

    #test = read_c3d_file \

    '''write_test_seq_in_binary('/home/taras/Documents/Datasets/MoCap/C3d/Test_seq/01_05.c3d',
                             FLAGS.data_dir + '/climb.binary')
    write_test_seq_in_binary('/home/taras/Documents/Datasets/MoCap/C3d/Test_seq/06_02.c3d',
                             FLAGS.data_dir + '/basketball.binary')
    write_test_seq_in_binary('/home/taras/Documents/Datasets/MoCap/C3d/Test_seq/14_01.c3d',
                             FLAGS.data_dir + '/boxing.binary')'''

    #test = read_c3d_file('/home/taras/Documents/Datasets/MoCap/Raw/train/14/14_10.c3d')

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
        write_binary()
    #result = read_test_seq_from_binary(FLAGS.data_dir + '/test_seq.binary')
    #print(test.shape)
