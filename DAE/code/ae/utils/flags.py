from __future__ import division
import os
from os.path import join as pjoin

import sys

import tensorflow as tf


def home_out(path):
  return pjoin(os.environ['HOME'], 'tmp', 'MoCap', path)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags about the sequence processing
flags.DEFINE_integer('chunk_length', 64, 'Length of the chunks, in which we will be processing our data. Define the length of the memory for RNN.')
flags.DEFINE_integer('chunking_stride', 64,'Stride for spliting sequences into the chunks')

# FLAGS about recurrency
flags.DEFINE_integer('recurrent_layer', 8,'At which layer we are going to apply recurrency')

# Autoencoder Architecture Specific Flags
flags.DEFINE_integer('DoF', 129, 'Dimensionality of the single frame')
flags.DEFINE_boolean('Hierarchical', False,
                     'Whether AE is hierarchical')
  

""" 							HIRERARCHICAL AE 			"""

# First encoding layer characteristics
flags.DEFINE_integer('chest_head_neurons', 10,
                     'Number of neurons for representation of the chest in hidden layer 1.')
flags.DEFINE_integer('right_arm_neurons', 20,
                     'Number of neurons for representation of the right arm in hidden layer 1.')
flags.DEFINE_integer('left_arm_neurons', 20,
                     'Number of neurons for representation of the left arm hidden layer 1.')
flags.DEFINE_integer('right_leg_neurons', 10,
                     'Number of neurons for representation of the right leg in hidden layer 1.')
flags.DEFINE_integer('left_leg_neurons', 10,
                     'Number of neurons for representation of the left leg in hidden layer 1.')
 
# Second encoding layer characteristics
flags.DEFINE_integer('spine_and_r_arm_neurons', 15,
                     'Number of neurons for representation of the trunk combined with right arm in hidden layer 2.')
flags.DEFINE_integer('spine_and_l_arm_neurons', 15,
                     'Number of neurons for representation of the trunk combined with left arm hidden layer 2.')
flags.DEFINE_integer('spine_and_r_leg_neurons', 15,
                     'Number of neurons for representation of the trunk combined with right leg in hidden layer 2.')
flags.DEFINE_integer('spine_and_l_leg_neurons', 15,
                     'Number of neurons for representation of the trunk combined with left leg in hidden layer 2.')
 
# Third encoding layer characteristics
flags.DEFINE_integer('upper_body_neurons', 10,
                     'Number of neurons for representation of the upper body in hidden layer 3.')
flags.DEFINE_integer('lower_body_neurons', 5,
                     'Number of neurons for representation of the lower body in hidden layer 3.')
 
# Third encoding layer characteristics
flags.DEFINE_integer('representation_size', 16,
                     'Number of neurons for representation of the whole body in hidden layer 4.')




""" 							FLAT AE 			"""

flags.DEFINE_integer("num_hidden_layers",5, "Number of hidden layers")

flags.DEFINE_integer('hidden1_units', 160,
                     'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2_units', 60,
                     'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3_units', 20,
                     'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4_units', 60,
                     'Number of units in hidden layer 4.') 
flags.DEFINE_integer('hidden5_units', 160,
                     'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6_units', 70,
                     'Number of units in hidden layer 6.') 



""" 							Training characteristics 			"""


flags.DEFINE_float('pretraining_learning_rate', 0.002,
                   'Initial learning rate.')
flags.DEFINE_float('training_learning_rate', 0.001,
                   'Initial learning rate.')
flags.DEFINE_float('learning_rate_decay', 0.7,
                   'Learning rate decaying factor.')


flags.DEFINE_float('variance_of_noise', 0.3, 'Coefficient to be multiplyied on a standart deviation of the data for the gaussian noise added to every point in input during the training')


# Constants
flags.DEFINE_integer('seed', 12345, 'Random seed')

flags.DEFINE_float('dropout', 0.95, 'Probability to keep the neuron on')

flags.DEFINE_integer('test_sequences', 50,
                     'Amount of the testing sequences.Each with the length from flag "chunk_length"')

flags.DEFINE_integer('validation_sequences', 50,
                     'Amount of the validation sequences. Each with the length from flag "chunk_length"')

flags.DEFINE_integer('batch_size', 12,
                     'Size of the mini batch')

flags.DEFINE_integer('pretraining_epochs', 1000,
                     "Number of training epochs for pretraining layers")
flags.DEFINE_integer('training_epochs', 100, #60 originaly
                     "Number of training epochs for pretraining layers")

flags.DEFINE_integer('middle_layer', 2,
                     "Which hidden layer is view as a middle layer with the representation")

flags.DEFINE_float('zero_bound', 1.0e-9,
                   'Value to use as buffer to avoid '
                   'numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                   'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')

flags.DEFINE_integer('amount_of_subfolders', 1, 'Amount of subfolders in the folder with the CMU MoCap dataset') # should be much more

# Directories
flags.DEFINE_string('data_dir','/home/taras/storage/data(daz)',
#flags.DEFINE_string('data_dir','/storage/taras/CMU',
                    'Directory to put the training data.')

flags.DEFINE_string('model_dir', '/home/taras/storage/MoCap/models',
                    'Directory to put the summary data')

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

# ADDITIONAL Flags
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
