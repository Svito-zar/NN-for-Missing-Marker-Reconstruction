


import tensorflow as tf
import numpy as np
import sys

from utils.data import read_c3d_file
from utils.flags import FLAGS

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_Tf_records(train_filename, sequence):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    # Create a feature
    feature = {'train/sequence': _bytes_feature(tf.compat.as_bytes(sequence.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

def read_Tf_records(data_path):
    with tf.Session() as sess:
        feature = {'train/sequence':  tf.FixedLenFeature([], tf.string)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=10)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        sequence_raw = tf.decode_raw(features['train/sequence'], tf.float32)

        # Reshape image data into the original shape
        sequence = tf.reshape(sequence_raw, [301,129])

        min_after_dequeue = 1
        capacity = min_after_dequeue + 3 * FLAGS.batch_size

        # Creates batches by randomly shuffling tensors
        sequences_batches = tf.train.shuffle_batch([sequence], batch_size=15, capacity=capacity, num_threads=1,
                                                min_after_dequeue=1)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)


        print(sequence.set_shape([301,129]))

        try:
            while not coord.should_stop():
                # Run training steps or whatever
                seq = sess.run(sequences_batches)
                print(seq)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    curr_sequence = read_c3d_file('/home/taras/Documents/Datasets/MoCap/C3d/train/01/01_04.c3d')

    #print(curr_sequence)

    # Move to TfRecords
    write_Tf_records('train.tfrecords',curr_sequence)

    read_Tf_records('train.tfrecords')

#NEXT TEST

'''tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6
X[1,6,:] = 0
X_lengths = [10, 6]

cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(64)
cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.5)
cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float32,
    sequence_length=X_lengths,
    inputs= tf.cast(X, tf.float32))

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)


print(result[0]["outputs"].shape)
print(result[0]["outputs"])
assert result[0]["outputs"].shape == (2, 10, 64)

# Outputs for the second example past past length 6 should be 0
assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()

print(result[0]["last_states"][0])'''



#NEXT TEST

'''from __future__ import division
#from __future__ import print_function

import tensorflow as tf
from utils.flags import FLAGS

prob_of_missing = 0.9

if(FLAGS.frame_size%3 != 0):
    print('ERROR: Frame size must be divisible by 3')
    exit(1)

random_size = [FLAGS.batch_size, FLAGS.chunk_length, int(FLAGS.frame_size/3)]
tensor_size = [FLAGS.batch_size, FLAGS.chunk_length, FLAGS.frame_size]


# Make sure that all coordinates of each point are either missing or present
random_missing_points = tf.random_uniform(random_size)
stacked_coords = tf.stack([random_missing_points, random_missing_points, random_missing_points], axis =3)
random_missing_coords = tf.reshape(stacked_coords, [tf.shape(stacked_coords)[0],tf.shape(stacked_coords)[1], -1])


mask = tf.where(random_missing_coords < 1 - prob_of_missing,
                tf.ones(tensor_size), tf.zeros(tensor_size))

input = tf.random_uniform(tensor_size)

output = tf.multiply(input, mask)

with tf.Session(''):

    values = output.eval()

    print values'''