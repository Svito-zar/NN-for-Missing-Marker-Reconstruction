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

#                       Flags about the sequence processing

flags.DEFINE_integer('chunk_length', 64, 'Length of the chunks, in which we will be processing our data. Define the length of the memory for RNN.')
flags.DEFINE_integer('chunking_stride', 64,'Stride for spliting sequences into the chunks')
flags.DEFINE_bool('reccurent', True, 'Whether AE is recurrent')

#                               Flags about training
flags.DEFINE_float('learning_rate', 0.0002 ,
                   'learning rate for training .')

flags.DEFINE_float('variance_of_noise', 0.3, 'Coefficient to be multiplyied on a standart deviation of the data for the gaussian noise added to every point in input during the training')

flags.DEFINE_boolean('restore', False,' Whether we restore the model from the checkpoint')
flags.DEFINE_integer('chkpt_num' , 26920, 'Number of the checkpoint')

flags.DEFINE_boolean('evaluate', False,' Whether we are evaluating the system or optimizing a hyper-parameter')

flags.DEFINE_float('dropout', 0.9, 'Probability to keep the neuron on')

flags.DEFINE_integer('batch_size', 32,
                     'Size of the mini batch')

flags.DEFINE_integer('training_epochs', 100,
                     "Number of training epochs for pretraining layers")

flags.DEFINE_float('Weight_decay', 0.005,' Whether we apply weight decay')

flags.DEFINE_boolean('Early_stopping',True,' Whether we do early stopping')
flags.DEFINE_float('delta_for_early_stopping', 0.5, 'How much worst the results must get in order for training to be terminated. 0.05 mean 5% worst than best we had. It did not help to apply early stopping')

#                       Autoencoder Architecture Specific Flags
flags.DEFINE_integer('frame_size', 123, 'Dimensionality of the input for a single frame')
flags.DEFINE_integer('amount_of_frames_as_input', 1, 'Amount of frames used as input at each time step')

flags.DEFINE_integer("num_hidden_layers",1,"Number of hidden layers")
flags.DEFINE_integer('network_width', 1024, 'Number of units in each hidden layer ')

#                       Flags about the missing markers
flags.DEFINE_boolean('missing_markers_are_random', True,' Whether we having missing markers at random positions, or the whole body part')
flags.DEFINE_float('missing_rate', 0.2, 'What fraction of the input is missing during the testing')
flags.DEFINE_float('defaul_value', 0, 'What number we will put in place of the missing marker')

#                           Constants

flags.DEFINE_integer('seed', 123456, 'Random seed')


#                           Directories

flags.DEFINE_string('data_dir',#'/home/taras/Documents/storage/CMU_Mocap/C3D/30k_90L',
                    #'/home/taras/Documents/Datasets/MoCap/C3d/30K',
		     '/home/taras/Documents/Datasets/MoCap/C3d/Raw/30k_90L',
                    'Directory to put the training data.')

flags.DEFINE_string('model_dir', home_out('models'),
                    'Directory to put the summary data')

flags.DEFINE_string('params_file',home_out('params'),
                    'File for saving the parameters values')

flags.DEFINE_string('summary_dir', home_out('summaries_exp'), 
                    'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', home_out('chkpts_exp'),
                    'Directory to put the model checkpoints')

flags.DEFINE_string('results_file', home_out('results.txt'),
                    'Filey to put the experimental results')


                                # ADDITIONAL Flags
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")


###                      Flags related to the testing of continuous missing of particular markers
flags.DEFINE_bool("continuos_gap", True,
                  "test in a situation when some markers are missing over long period of time")
flags.DEFINE_integer('amount_of_missing_markers', 15, 'amount of markers which are going to be missing during the "real life" test')
flags.DEFINE_integer('duration_of_a_gap', 600, 'amount of frames of a gap: how long markers are going to be missing, if None - random uniform between 6 and 60')
flags.DEFINE_string('contin_test_file', 'our_model_results.txt',
                    'Filey to put the results of the test in "real life" conditions')
flags.DEFINE_bool("plot_error", False,
                  "plot error for every frame in a situation when some markers are missing over long period of time")
flags.DEFINE_integer('skip_duration', 180, 'amount of first frames to be ignored')
flags.DEFINE_integer('no_gap_duration', 120, 'amount of frames with all the markers at the beginning and end of the sequence')