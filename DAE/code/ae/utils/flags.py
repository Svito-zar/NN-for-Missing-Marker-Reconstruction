from __future__ import division
import os
from os.path import join as pjoin

import sys

import tensorflow as tf


def home_out(path):
  return pjoin(os.environ['HOME'], 'tmp', 'MoCap', path)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Autoencoder Architecture Specific Flags
flags.DEFINE_integer("num_hidden_layers", 3, "Number of hidden layers") # should be 3

flags.DEFINE_integer('hidden1_units', 256,
                     'Number of units in hidden layer 1.') # 2000 originaly
flags.DEFINE_integer('hidden2_units', 16,
                     'Number of units in hidden layer 2.') # 2000 originaly
flags.DEFINE_integer('hidden3_units', 64,
                     'Number of units in hidden layer 3.')

flags.DEFINE_integer('DoF', 132, 'Dimensionality of the single frame') # should be much more

# Maximal amount of hidden layers is defined by the last value 'pre_layer4_learning_rate' -> 4
flags.DEFINE_float('pretraining_learning_rate', 0.0003,
                   'Initial learning rate.')

# It is a question wheather adding noise 
flags.DEFINE_float('variance_of_noise', 0.1, 'Standart deviation of the gaussian noise added to every point in input')

# Constants
flags.DEFINE_integer('seed', 1234, 'Random seed')

flags.DEFINE_float('dropout', 0.8, 'Probability to keep the neuron on')

flags.DEFINE_integer('test_sequences_numb', 5,
                     'Amount of the testing sequences.')

flags.DEFINE_integer('batch_size', 1000,
                     'Size of the mini batch')

flags.DEFINE_float('supervised_learning_rate', 0.1,
                   'Supervised initial learning rate.')

flags.DEFINE_integer('pretraining_epochs', 4, #60 originaly
                     "Number of training epochs for pretraining layers")

flags.DEFINE_integer('middle_layer', 2,
                     "Which hidden layer is view as a middle layer with the representation")

flags.DEFINE_float('zero_bound', 1.0e-9,
                   'Value to use as buffer to avoid '
                   'numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                   'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')

flags.DEFINE_integer('amount_of_subfolders', 5, 'Amount of subfolders in the folder with the CMU MoCap dataset') # should be much more

# Directories
flags.DEFINE_string('data_dir','/home/taras/storage/data(daz)',
#flags.DEFINE_string('data_dir','/storage/taras/CMU',
                    'Directory to put the training data.')
flags.DEFINE_string('params_file','/home/taras/storage/MoCap/params',
                    'File for saving the parameters values')

flags.DEFINE_string('summary_dir', home_out('summaries'),
                    'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', home_out('chkpts'),
                    'Directory to put the model checkpoints')

# TensorBoard
flags.DEFINE_boolean('no_browser', True,
                     'Whether to start browser for TensorBoard')

# Python
flags.DEFINE_string('python', sys.executable,
                    'Path to python executable')
