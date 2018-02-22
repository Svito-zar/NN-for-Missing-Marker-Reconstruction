"""Functions for downloading,reading and preprocessing CMU data."""

from __future__ import division
from __future__ import print_function

import btk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from flags import *


class DataSet(object):
    '''
    A class for storing a dataset and all important information,
    which might be needed during training,
    such as batch size amount of epochs completed and so on.
    '''

    def __init__(self, sequences, batch_size):
        self._batch_size = batch_size
        self._sequences = sequences  # all the sequnces in the dataset
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

    def next_batch(self):
        """Return the next batch of sequences from this data set."""
        batch_numb = self._index_in_epoch
        self._index_in_epoch += self._batch_size
        if self._index_in_epoch > self._num_chunks:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_sequences)
            np.random.shuffle(perm)
            self._sequences = self._sequences[perm]
            # Start next epoch
            batch_numb = 0
            self._index_in_epoch = self._batch_size
        return self._sequences[batch_numb:batch_numb + self._batch_size:1, :]


class DataSets(object):
    '''
      A class for storing Train and Eval datasets and all related information,
      '''
    pass


def read_c3d_file(file_name):
    """
    Reads a file from CMU MoCap dataset in the c3d format

       Returns:
            sequence [sequence_length,frame_size] - local channels transformed to the hips-centered coordinates
            hips [frame_size]                     - coordinates of the hips
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

    all_3d_coords = np.array([])
    point_id = 0

    # Get subject name
    name = ''
    for i in range(0, acq.GetPoints().GetItemNumber()):
        label_string = acq.GetPoint(i).GetLabel()
        name_parts = len(label_string.split(':', 2))
        if debug:
            print(label_string.split(':', 2))
        if (name_parts == 2 and label_string.split(':', 2)[1] == "LSHO"):
            name = label_string.split(':', 2)[0] + ':'

    if debug:
        print(name)

    # Check scaling factor
    file_instance = btk.btkC3DFileIO()
    file_instance.Read(file_name, acq)

    if debug:
        print("Units:", acq.GetPointUnit())

    missing_markers = list()

    while (True):
        try:
            # Get the next label
            label = name + labels_file.readline().splitlines()[0]
            if debug and point_id < 7:
                print(label)
            next_point = acq.GetPoint(label).GetValues()
        except IndexError:
            print('Read', point_id, 'skeleton 3d points during', acq.GetPointFrameNumber(), 'time frames')
            break

        # Check if the data is there
        if next_point[0][0] == 0 and next_point[0][1] == 0 and next_point[0][2] == 0:
            print('Problematic label was ', label)
            print('Last: ', next_point[-1])
            for time_step in range(next_point.shape[0]):
                if next_point[time_step][0] == 0:
                    missing_markers.append(point_id)

        # Concatanate curr chunks to all of them
        all_3d_coords = np.dstack([all_3d_coords, next_point]) if all_3d_coords.size else np.array(next_point)

        point_id += 1

    print(missing_markers)

    # Convert to the hips centered coordinates
    hips = np.zeros([all_3d_coords.shape[0], 3])  # coordinates of the hips at each time-step

    for time_step in range(all_3d_coords.shape[0]):
        # Obtain hips coords
        hips_x = np.average(all_3d_coords[time_step][0][23:27])
        hips_y = np.average(all_3d_coords[time_step][1][23:27])
        hips_z = np.average(all_3d_coords[time_step][2][23:27])
        hips[time_step] = [hips_x, hips_y, hips_z]

        # Subtract hips coors from each joint
        all_3d_coords[time_step] -= hips[time_step][:, np.newaxis]

    amount_of_frames = acq.GetPointFrameNumber()

    # Replace missing markers with the same marker at the end of the sequence
    for marker_id in missing_markers:
        for coord in range(3):
            if marker_id > 22 and marker_id < 27:  # if that are the hips
                all_3d_coords[time_step][coord][marker_id] = 0  # center - hips coords
            else:
                all_3d_coords[time_step][coord][marker_id] = all_3d_coords[amount_of_frames - 1][coord][marker_id]

    # Make a proper array shape
    mocap_seq = all_3d_coords.reshape(all_3d_coords.shape[0], -1)  # Concatanate all coords into one vector

    # Calculate max_val
    max = np.amax(np.absolute(mocap_seq))
    if max > 100000:
        print("\nWATCH! The file ", file_name, " had maximal value ", max, "\n")

    # For debug - Visualize the skeleton
    # visualize(mocap_seq)

    return mocap_seq


def read_unlabeled_data(train_dir, evaluate):
    """
      Reads all 3 datasets from CMU MoCap dataset in C3D format

      Args:
          train_dir - address to the train, dev and eval datasets
          evaluate - flag : weather we want to evaluate a network or we just optimize hyperparameters
      Returns:
          datasets - object of class DataSets, containing Train and Eval datasets
          max_val - maximal value in the raw data ( for post-processing)
          mean_pose - mean pose in the raw data ( for post-processing)
    """

    data_sets = DataSets()

    # Get constants from the file
    data_dir = FLAGS.data_dir
    chunk_length = FLAGS.chunk_length
    stride = FLAGS.chunking_stride

    if stride > chunk_length:
        print(
            'ERROR! \nYou have stride bigger than lentgh of chunks. '
            'Please, change those values at flags.py, so that you don\'t ignore the data')
        exit(0)

    # #########             Get TRAIN data                  ###########
    print('\nReading train data from the following folder ... ', data_dir + '/train')

    train_data = read_a_folder(data_dir + '/train')

    [amount_of_train_strings, seq_length, DoF] = train_data.shape
    print('\n' + str(amount_of_train_strings) + ' sequences with length ' + str(
        seq_length) + ' will be used for training')

    #         #########             Get TEST data                  ###########

    print('\nReading test data from the following folder : ', data_dir + '/eval')
    if evaluate:
        test_data = read_a_folder(data_dir + '/eval')
    else:
        test_data = read_a_folder(data_dir + '/dev')

    [amount_of_test_strings, seq_length, DoF] = test_data.shape
    print(
        '\n' + str(amount_of_test_strings) + ' sequences with length ' + str(seq_length) + ' will be used for testing')

    # Do mean normalization : substract mean pose
    mean_pose = train_data.mean(axis=(0, 1))
    train_data = train_data - mean_pose[np.newaxis, np.newaxis, :]
    test_data = test_data - mean_pose[np.newaxis, np.newaxis, :]

    # Scales all values in the input_data to be between -1 and 1
    eps = 1e-8
    max_train = np.amax(np.absolute(train_data), axis=(0, 1))
    max_test = np.amax(np.absolute(test_data), axis=(0, 1))
    max_val = np.maximum(max_train, max_test)
    train_data = np.divide(train_data, max_val[np.newaxis, np.newaxis, :] + eps)
    test_data = np.divide(test_data, max_val[np.newaxis, np.newaxis, :] + eps)

    # Check the data range
    max_ = test_data.max()
    min_ = test_data.min()

    print("MAximum value in the normalized test dataset : " + str(max_))
    print("Minimum value in the normalized test dataset : " + str(min_))

    print('\nTrain data shape: ', train_data.shape)

    data_sets.train = DataSet(train_data, FLAGS.batch_size)
    data_sets.test = DataSet(test_data, FLAGS.batch_size)

    # Assign variance
    data_sets.train.sigma = np.std(train_data, axis=(0, 1))

    # Check if we have enough data
    if data_sets.train._num_sequences < data_sets.train._batch_size:
        print('ERROR: We have got not enough data! Reduce batch_size or increase amount of subfolder you use.')
        exit(1)

    return data_sets, max_val, mean_pose


def read_a_folder(curr_dir):
    chunk_length = FLAGS.chunk_length
    stride = FLAGS.chunking_stride

    data = np.array([])

    for sub_dir in os.listdir(curr_dir):
        for filename in os.listdir(curr_dir + '/' + sub_dir):
            curr_sequence = read_c3d_file(curr_dir + '/' + sub_dir + '/' + filename)

            curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in
                                    xrange(0, len(curr_sequence) - chunk_length, stride)])  # Split sequence into chunks

            if curr_chunks.shape[0] > 0:
                # Concatanate curr chunks to all of them
                data = np.vstack([data, curr_chunks]) if data.size else np.array(curr_chunks)

        print(data.shape)

        if data.shape[0] > 50000:
            break

    data = np.array(data)

    return data


def add_noise(x, variance_multiplier, sigma):
    """
           Add Gaussian noise to the data
           Args:
               x                   - input vector
               variance_multiplier - coefficient to multiple on, when we calculate a variance of the noise
               sigma               - variance of the dataset
           Returns:
               x - output vector, noisy data
    """
    eps = 1e-15
    noise = tf.random_normal(x.shape, 0.0, stddev=np.multiply(sigma, variance_multiplier) + eps)
    x = x + noise
    return x


def read_dataset_and_write_in_binary(evaluate):
    """
              Reads 3 datasets: "Train","Dev" and "Eval" from the CMU MoCap dataset in bvh format
              And write them in the binary format.
              Will get the address of the folder with the data from flags.py
              Args:
                  evaluate - flag, indicating weather we are going to evaluate the system or we are optimizing hyper-parameters
              Returns:
                  will write binary files in the same folder as the original data
    """

    # Get the data
    data, max_val, mean_pose = read_unlabeled_data(FLAGS.data_dir, False)  # read_all_the_data()

    # Write all important information into binary files

    # Datasets themselfs
    train_file = open(FLAGS.data_dir + '/train.binary', 'wb')
    data.train._sequences.tofile(train_file)
    train_file.close()

    eval_file = open(FLAGS.data_dir + '/eval.binary', 'wb')
    data.test._sequences.tofile(eval_file)
    eval_file.close()

    # Dataset properties

    sigma_file = open(FLAGS.data_dir + '/variance.binary', 'wb')
    data.train.sigma.tofile(sigma_file)
    sigma_file.close()

    max_val_file = open(FLAGS.data_dir + '/maximums.binary', 'wb')
    max_val.tofile(max_val_file)
    max_val_file.close()

    mean_file = open(FLAGS.data_dir + '/mean.binary', 'wb')
    mean_pose.tofile(mean_file)
    mean_file.close()

    print('All the binary files for the dataset was saved in the folder ', FLAGS.data_dir)


def read_binary_dataset(dataset_name):
    filename = FLAGS.data_dir + '/' + dataset_name + '.binary'
    dataset = np.fromfile(filename)
    amount_of_frames = int(dataset.shape[0] / (FLAGS.chunk_length * FLAGS.frame_size * FLAGS.amount_of_frames_as_input))
    # Clip array so that it divides exactly into the inputs we want (frame_size * amount_of_frames_as_input)
    dataset = dataset[0:amount_of_frames * FLAGS.chunk_length * FLAGS.frame_size * FLAGS.amount_of_frames_as_input]
    # Reshape
    dataset = dataset.reshape(amount_of_frames, FLAGS.chunk_length, FLAGS.frame_size * FLAGS.amount_of_frames_as_input)
    return dataset


def read_datasets_from_binary():
    """
      Reads train and test datasets and their properties from binary file format

      Will take them from the corresponding file in the folder, which is defined by FLAGS.data_dir

      Returns:
          datasets  - object of class DataSets, containing Train and Eval datasets
          max_val   - maximal value in the raw data ( for post-processing)
          mean_pose - mean pose in the raw data ( for post-processing)

    """
    data_sets = DataSets()

    #         #########             Get TRAIN data                  ###########

    train_data = read_binary_dataset('train')
    [amount_of_train_strings, seq_length, DoF] = train_data.shape
    print('\n' + str(amount_of_train_strings) + ' sequences with length ' + str(seq_length) + ' and ' + str(
        FLAGS.amount_of_frames_as_input) + ' frames in each will be used for training')

    #         #########             Get TEST data                  ###########

    test_data = read_binary_dataset('eval')
    [amount_of_test_strings, seq_length, DoF] = test_data.shape
    print(str(amount_of_test_strings) + ' sequences with length ' + str(seq_length) + ' will be used for testing')

    # Shuffle the data
    perm = np.arange(amount_of_train_strings)
    np.random.shuffle(perm)
    train_data = train_data[perm]

    data_sets.train = DataSet(train_data, FLAGS.batch_size)
    data_sets.test = DataSet(test_data, FLAGS.batch_size)

    # Assign variance
    data_sets.train.sigma = np.std(train_data, axis=(0, 1))

    # Read maximal value and mean pose before normalizatio
    max_val = np.fromfile(FLAGS.data_dir + '/maximums.binary')
    mean_pose = np.fromfile(FLAGS.data_dir + '/mean.binary')

    # Check if we have enough data
    if data_sets.train._num_sequences < data_sets.train._batch_size:
        print('ERROR: We have got not enough data! Reduce batch_size or increase amount of subfolder you use.')
        exit(1)

    return data_sets, max_val, mean_pose


def write_test_seq_in_binary(input_file_name, output_file_name):
    """ Read test sequence in c3d format and
        write it into the binart file

      Args:
        input_file_name:  the name of the input file
        output_file_name: the name of the output file
      Returns:
        nothing
    """
    test_file = open(output_file_name, 'wb')
    test_seq = read_c3d_file(input_file_name)
    test_seq.tofile(test_file)
    test_file.close()
    print("The test sequence was read from", input_file_name, " and written to", output_file_name)


def read_test_seq_from_binary(binary_file_name):
    """ Read test sequence from the binart file

          Args:
            binary_file_name:  the name of the input binary file
          Returns:
            read_seq:          test sequence
    """
    # Read the sequence
    read_seq = np.fromfile(binary_file_name)
    # Reshape
    read_seq = read_seq.reshape(-1, FLAGS.frame_size)
    amount_of_frames = int(read_seq.shape[0] / (FLAGS.chunk_length * FLAGS.amount_of_frames_as_input))
    if amount_of_frames > 0:
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
        error_scaled = tf.multiply(error, max_vals[np.newaxis, :] + 1e-15)
        squared_error = tf.reduce_mean(tf.square(error_scaled))
        return squared_error


def visualize(mocap_seq, test=False):
    all_3d_coords = mocap_seq.reshape(-1, 3, 41)  # Concatanate all coords into one vector

    # For debug - Visualize the skeleton
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_frame = 40
    treshhold_0 = 14
    treshhold_1 = 20
    treshhold_2 = 27
    coef = 100
    for step in range(start_frame, start_frame + 30, 10):

        # Visualize a 3D point cloud
        ax.scatter3D(all_3d_coords[step][0][:treshhold_0],
                     np.add(all_3d_coords[step][1][:treshhold_0], (step - start_frame) * coef),
                     all_3d_coords[step][2][:treshhold_0], c='c', marker='o')
        ax.scatter3D(all_3d_coords[step][0][treshhold_0:treshhold_1],
                     np.add(all_3d_coords[step][1][treshhold_0:treshhold_1], (step - start_frame) * coef),
                     all_3d_coords[step][2][treshhold_0:treshhold_1], c='r', marker='o')
        ax.scatter3D(all_3d_coords[step][0][treshhold_1:treshhold_2],
                     np.add(all_3d_coords[step][1][treshhold_1:treshhold_2], (step - start_frame) * coef),
                     all_3d_coords[step][2][treshhold_1:treshhold_2], c='y', marker='o')
        ax.scatter3D(all_3d_coords[step][0][treshhold_2:],
                     np.add(all_3d_coords[step][1][treshhold_2:], (step - start_frame) * coef),
                     all_3d_coords[step][2][treshhold_2:], c='b', marker='o')

        # Find which points are present

        key_point_arm = []
        for point in list([0, 1, 2, 7, 8, 9]):
            if all_3d_coords[step][0][point] != 0 and all_3d_coords[step][0][point + 1] != 0:
                if all_3d_coords[step][1][point] != 0 and all_3d_coords[step][1][point + 1] != 0:
                    if all_3d_coords[step][2][point] != 0 and all_3d_coords[step][2][point + 1] != 0:
                        key_point_arm.append(point)
        key_point_arm = np.array(key_point_arm)

        #print(key_point_arm)

        key_point_leg = []
        for point in list([27, 34]):  # 28, 35
            if all_3d_coords[step][0][point] != 0 and all_3d_coords[step][0][point + 1] != 0:
                if all_3d_coords[step][1][point] != 0 and all_3d_coords[step][1][point + 1] != 0:
                    if all_3d_coords[step][2][point] != 0 and all_3d_coords[step][2][point + 1] != 0:
                        key_point_leg.append(point)
        key_point_leg = np.array(key_point_leg)

        # Add lines in between

        for point in key_point_arm:
            xline = all_3d_coords[step][0][point:point + 2]
            yline = np.add(all_3d_coords[step][1][point:point + 2], (step - start_frame) * coef)
            zline = all_3d_coords[step][2][point:point + 2]
            ax.plot(xline, yline, zline, c='c')
        for point in key_point_leg:
            xline = all_3d_coords[step][0][point:point + 3:2]
            yline = np.add(all_3d_coords[step][1][point:point + 3:2], (step - start_frame) * coef)
            zline = all_3d_coords[step][2][point:point + 3:2]
            ax.plot(xline, yline, zline, c='b')

    plt.show()


if __name__ == '__main__':

    # Do some testing

    Test = False

    if Test:
        write_test_seq_in_binary(FLAGS.data_dir + '/../test_seq/86_14.c3d',
                                 FLAGS.data_dir + '/basketball_2.binary')
        write_test_seq_in_binary(FLAGS.data_dir + '/../test_seq/85_02.c3d',
                                 FLAGS.data_dir + '/salto.binary')
        write_test_seq_in_binary(FLAGS.data_dir + '/../test_seq/14_01.c3d',
                                 FLAGS.data_dir + '/boxing.binary')

    else:
        read_dataset_and_write_in_binary(True)
