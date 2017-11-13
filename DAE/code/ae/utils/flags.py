from __future__ import division
import os
from os.path import join as pjoin
import sys

import tensorflow as tf

def home_out(path):
  return pjoin(os.environ['HOME'], 'tmp', 'MoCap', path)

flags = tf.app.flags
FLAGS = flags.FLAGS

"""  							Fine-tuning Parameters 				"""

# Flags about the sequence processing

flags.DEFINE_integer('chunk_length', 1, 'Length of the chunks, in which we will be processing our data. Define the length of the memory for RNN.')
flags.DEFINE_integer('chunking_stride', 1,'Stride for spliting sequences into the chunks')
flags.DEFINE_bool('reccurent', False, 'Whether AE is recurrent')

flags.DEFINE_float('learning_rate', 0.00004 ,
                   'learning rate for training .')

flags.DEFINE_float('variance_of_noise', 0.00, 'Coefficient to be multiplyied on a standart deviation of the data for the gaussian noise added to every point in input during the training')
# Constants

flags.DEFINE_integer('seed', 123456, 'Random seed')

flags.DEFINE_float('dropout', 1.0, 'Probability to keep the neuron on')

flags.DEFINE_integer('batch_size', 128,
                     'Size of the mini batch')

flags.DEFINE_integer('pretraining_epochs',5,
                     "Number of training epochs for pretraining layers")
flags.DEFINE_integer('training_epochs', 100,
                     "Number of training epochs for pretraining layers")

# Autoencoder Architecture Specific Flags
flags.DEFINE_integer('frame_size', 96, 'Dimensionality of the input for a single frame')
flags.DEFINE_integer('amount_of_frames_as_input', 10, 'Amount of frames used as input at each time step')

flags.DEFINE_boolean('Hierarchical', False,
                     'Whether AE is hierarchical')

# Flags about the missing markers
flags.DEFINE_boolean('missing_markers_are_random', False ,' Whether we having missing markers at random positions, or the whole right hand')
flags.DEFINE_float('missing_rate', 0.2, 'What fraction of the input is missing during the testing')
flags.DEFINE_float('defaul_value', 0, 'What number we will put in place of the missing marker')

# Flags about training
flags.DEFINE_boolean('restore', True,' Whether we restore the model from the checkpoint')
flags.DEFINE_integer('chkpt_num' , 1016735, 'Number of the checkpoint')

flags.DEFINE_boolean('evaluate', False,' Whether we are evaluating the system or optimizing a hyper-parameter')

flags.DEFINE_boolean('Layer_wise_Pretraining', False,' Whether we do layer-wise pretraining')

flags.DEFINE_integer('Weight_decay', None,' Whether we apply weight decay') 
flags.DEFINE_boolean('Early_stopping',False,' Whether we do early stopping')

flags.DEFINE_float('delta_for_early_stopping', 1.5, 'How much worst the results must get in order for training to be terminated. 0.05 mean 5% worst than best we had')


""" 							FLAT AE 			"""

flags.DEFINE_integer("num_hidden_layers",1,"Number of hidden layers")
flags.DEFINE_integer('network_width', 2048, 'Number of units in each hidden layer ')


""" 							Other parameters			"""

flags.DEFINE_integer('middle_layer', 1,
                     "Which hidden layer is view as a middle layer with the representation")

flags.DEFINE_float('zero_bound', 1.0e-9,
                   'Value to use as buffer to avoid '
                   'numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9,
                   'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')


# Directories

flags.DEFINE_string('data_dir',#'/home/taras/storage/data(daz)',
                    '/home/taras/Documents/Datasets/MoCap/BVH/Raw',
                    #'/home/taras/Documents/Datasets/CMU_c3d/Walk',
                     #'/home/taras/Dropbox/Taras/2017_PhD_at_KTH/Datasets/CMU_c3d',
                    'Directory to put the training data.')

flags.DEFINE_string('model_dir', '/home/taras/storage/MoCap/models',
                    'Directory to put the summary data')

flags.DEFINE_string('params_file','/home/taras/storage/MoCap/params',
                    'File for saving the parameters values')

flags.DEFINE_string('summary_dir', home_out('summaries_exp'), 
                    'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', home_out('chkpts_exp'),
                    'Directory to put the model checkpoints')

flags.DEFINE_string('results_file', home_out('results.txt'),
                    'Filey to put the experimental results')


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
