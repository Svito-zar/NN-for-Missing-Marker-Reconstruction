from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from utils.data import read_all_the_data, read_c3d_file, read_test_seq_from_binary
from utils.flags import FLAGS
# import class for both architectures of AE
from FlatAE import FlatAutoEncoder, loss_reconstruction
from HierarchicalAE import HierarchicalAE
from AE import use_existing_markers,simulate_missing_markets

class DataInfo(object):
  """Information about the datasets

   Will be passed to the FlatAe for creating corresponding variables in the graph
  """

  def __init__(self, data_sigma, train_shape, eval_shape, max_val):
    """DataInfo initializer

    Args:
      data_sigma:   variance in the dataset
      train_shape:  dimensionalities of the train dataset
      test_shape:  dimensionalities of the testing dataset
      eval_shape:  dimensionalities of the evaluation dataset
    """
    self._data_sigma = data_sigma
    self._train_shape =  train_shape
    self._eval_shape = eval_shape
    self._max_val = max_val

def learning(restore, data, max_val, learning_rate, batch_size, dropout):
  """ Unsupervised training of the autoencoder

  Returns:
    Autoencoder trained on a data provided by FLAGS
  """

    
  with tf.Graph().as_default() as g:

    start_time = time.time()

    # Read the flags
    chunk_length = int(64*1.0/ FLAGS.amount_of_frames_as_input)
    pretrain = FLAGS.Layer_wise_Pretraining
    variance = FLAGS.variance_of_noise

    #Get the information about the dataset
    data_info = DataInfo( data.train.sigma, data.train._sequences.shape, data.test._sequences.shape, max_val)

    # Allow tensorflow to change device alocation when needed
    config = tf.ConfigProto(allow_soft_placement=True) # log_device_placement=True)
    # Adjust configuration so that multiple execusions are possible
    config.gpu_options.allow_growth = True
    # Start a session
    sess = tf.Session(config=config)
    
    # Get variables from flags
    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes =[FLAGS.network_width # [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                            for j in xrange(num_hidden)]
    
    if(FLAGS.Hierarchical):
        if (FLAGS.Layer_wise_Pretraining):
            print('\nERROR! We cannot do layerwise pretraining for the hierarchical AE! Please, change the flags')
            exit(1)
        if (FLAGS.amount_of_frames_as_input > 1):
            print('\nERROR! We cannot take a few frames as an input for the hierarchical AE! Please, change the flags')
            exit(1)
        ae = HierarchicalAE( FLAGS.network_width, FLAGS.num_hidden_layers, FLAGS.variance_of_noise, data_info, sess)
        print('\nHierarchical AE was created !')

    else:
        # Get variables from flags
        num_hidden = FLAGS.num_hidden_layers
        ae_hidden_shapes =[FLAGS.network_width # [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                                for j in xrange(num_hidden)]

        # Check if the middle layer exists
        if(FLAGS.middle_layer > FLAGS.num_hidden_layers):
            print("ERROR: middle layer cannot be more than the total amount of layers! Please, change flags accordingly")
            exit(1)

        #Check if recurrency is set in the correct way
        if (FLAGS.reccurent==False and FLAGS.chunk_length > 1):
            print("ERROR: Without recurrency chunk length should be 1! Please, change flags accordingly")
            exit(1)

        ae_shape = [FLAGS.frame_size * FLAGS.amount_of_frames_as_input] + ae_hidden_shapes + [FLAGS.frame_size * FLAGS.amount_of_frames_as_input] # Do not take fingers as an input

        # Create an autoencoder
        ae  = FlatAutoEncoder(ae_shape, sess, batch_size, variance,data_info)
        print('\nFlat AE was created : ', ae_shape)


    if(restore and pretrain):
      print('ERROR! You cannot restore and pretrain at the same time! Please, chose one of these options')
      exit(1)


    # Check if the flags makes sence
    if(dropout < 0 or variance < 0):
      print('ERROR! Have got negative values in the flags!')
      exit(1)

    sess.run(tf.local_variables_initializer()) # to initialize input_producer

    with tf.variable_scope("Train") as main_scope:

        ##############        DEFINE  Optimizer and training OPERATOR      ####################################

         # Define optimizers
        optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate) #could be FLAGS.training_learning_rate
            
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ae._loss, tvars),   1e12)
        train_op = optimizer.apply_gradients(zip(grads, tvars),  global_step = tf.contrib.framework.get_or_create_global_step())

        # Create a saver
        saver = tf.train.Saver()  # saver = tf.train.Saver(variables_to_save)

        # Prepare for making a summary for TensorBoard
        train_error =tf.placeholder(dtype=tf.float32, shape=(), name='train_error')
        eval_error =tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')
        test_error =tf.placeholder(dtype=tf.float32, shape=(), name='test_rmse')
        tf.summary.scalar('Train_error', train_error)
        train_summary_op = tf.summary.merge_all()
        eval_summary_op =  tf.summary.scalar('Validation_error',eval_error)
        test_summary_op =  tf.summary.scalar('Test_rmse', test_error)

        summary_dir = FLAGS.summary_dir
        summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())

        num_batches = int(data.train._num_chunks/batch_size)
        num_test_batches = int(data.test._num_chunks/batch_size) - 1

        # Initialize the part of the graph with the input data
        sess.run(ae._train_data.initializer,
                 feed_dict={ae._train_data_initializer: data.train._sequences})
        sess.run(ae._valid_data.initializer,
                 feed_dict={ae._valid_data_initializer: data.test._sequences})


        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if(FLAGS.Layer_wise_Pretraining):
          layers_amount = len(ae_shape) - 2

          # create an optimizers
          pretrain_optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)

          # Make an array of the trainers for all the layers
          trainers=[pretrain_optimizer.minimize(loss_reconstruction(ae.run_less_layers(ae._input_, i+1), ae.run_less_layers(ae._input_, i+1, is_target=True), max_val), global_step=tf.contrib.framework.get_or_create_global_step(), name='Layer_wise_optimizer_'+str(i)) for i in xrange(len(ae_shape) - 2)]

          # Initialize all the variables
          sess.run(tf.global_variables_initializer())
            
        else:
          # Initialize variables
          sess.run(tf.global_variables_initializer())

        #restore model, if needed
        if(restore):
          chkpt_file = FLAGS.chkpt_dir+'/chkpt-56730'
          saver.restore(sess, chkpt_file)
          print("Model restored from the file "+str(chkpt_file) + '.')

        # A few initialization for the early stopping
        delta = 0.08 # error tolerance for early stopping
        best_error = 10000
        num_valid_batches = int(data.test._num_chunks/batch_size)

        try:

          if(FLAGS.Layer_wise_Pretraining):
              for i in xrange(layers_amount):
                  n = i + 1
                  print('Pretraining layer number ', n,' ... ')
        
                  with tf.variable_scope("layer_{0}".format(n)):

                    layer = ae.run_less_layers(ae._input_, n)

                    with tf.name_scope("pretraining_loss"):
                      target_for_loss = ae.run_less_layers(ae._input_, n, is_target=True)

                    loss =  loss_reconstruction(layer, target_for_loss)
                    
                    pretrain_trainer = trainers[i]
                    
                    for steps in xrange(num_batches * FLAGS.pretraining_epochs):
                      
                      if(coord.should_stop()):
                        break
                        
                      loss_summary, loss_value  = sess.run([pretrain_trainer , loss], feed_dict={ae._keep_prob: dropout} )
                  
                  # Copy the trained weights to the fixed matrices and biases
                  ae[ae._weights_str.format(n) + 'fixed'] = ae._w(n)
                  ae[ae._biases_str.format(n) + 'fixed'] = ae._b(n)
              
              loss_summary, loss_value  = sess.run([train_op, ae._reconstruction_loss], feed_dict={ae._keep_prob: dropout, ae._mask: ae._mask_generator.eval(session=ae.session)} )
              train_error_ = loss_value
            
          step = 0
          
          # Train the whole network jointly
          print('\nWe train on ', num_batches, ' batches with ', batch_size, ' training examples in each for', FLAGS.training_epochs, ' epochs...')
          print("")
          print(" ______________ ______")
          print("|     Epoch    | RMSE |")
          print("|------------  |------|")

          while not coord.should_stop():

            loss_summary, loss_value  = sess.run([train_op, ae._reconstruction_loss], feed_dict={ ae._mask: ae._mask_generator.eval(session=ae.session)})
            # For FLAT I would add ae._keep_prob: dropout into the feed_dict
            train_error_ = loss_value

            if (step % num_batches == 0):
  	      rmse = test(ae, FLAGS.data_dir + '/../test_seq/boxing.binary', max_val, mean_pose)
  	      print("\nOur RMSE for boxing is : ", rmse)
              
              epoch = step * 1.0 / num_batches
              # Write summary
              train_summary = sess.run(train_summary_op, feed_dict={train_error: np.sqrt(train_error_)}) # provide a value for a tensor with a train value

              # Print results of screen
              epoch_str ="| {0:3.0f} ".format(epoch)[:5]
              percent_str = "({0:3.2f}".format(epoch * 100.0 / FLAGS.training_epochs)[:5]
              error_str = "%) |{0:5.2f}".format(train_error_)[:10] + "|"
              print(epoch_str, percent_str,error_str) #output)

              if (epoch > 0):
                  summary_writer.add_summary(train_summary, step)

                  #Evaluate on the validation sequences
                  error_sum=0
                  for valid_batch in range(num_valid_batches):
                    curr_err = sess.run([ae._valid_loss],feed_dict={ae._mask:  ae._mask_generator.eval(session=ae.session)})
                    error_sum += curr_err[0]
                  new_error = error_sum / (num_valid_batches)
                  eval_sum = sess.run(eval_summary_op, feed_dict={eval_error: np.sqrt(new_error)})
                  summary_writer.add_summary(eval_sum, step)

                  '''# Evaluate on the test sequence
                  rmse = test(ae, FLAGS.data_dir + '/boxing.binary', max_val, mean_pose)
                  test_sum = sess.run(test_summary_op, feed_dict={test_error: (rmse / ae.scaling_factor)})
                  summary_writer.add_summary(test_sum, step)'''

                  # Early stopping
                  if(FLAGS.Early_stopping):
                    if((new_error - best_error) / best_error > delta):
                      print('After '+str(step) + ' steps the training started over-fitting ')
                      break
                    if(new_error < best_error):
                      best_error = new_error

                      # Saver for the model
                      # curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                      save_path = saver.save(sess, FLAGS.chkpt_dir+'/chkpt', global_step=step) # `save` method will call `export_meta_graph` implicitly.

            step += 1
            
        except tf.errors.OutOfRangeError:
          if not FLAGS.Early_stopping:
            # Save for the model
            save_path = saver.save(sess, FLAGS.chkpt_dir+'/chkpt', global_step=step) # `save` method will call `export_meta_graph` implicitly.
          print('Done training for %d epochs, %d steps.' % (FLAGS.training_epochs, step))
          print("The final model was saved in file: %s" % save_path)
        finally:
          # When done, ask the threads to stop.
          coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

    duration = (time.time() - start_time)/ 60 # in minutes, instead of seconds

    print("The training was running for %.3f  min" % (duration))

    # SAve the results
    if (FLAGS.reccurent):
        f = open(FLAGS.results_file, 'a')
        f.write('\nRecurrent AE For the data from ' + str(FLAGS.data_dir) + ' Width: ' + str(
            FLAGS.network_width) + ' and depth : ' + str(FLAGS.num_hidden_layers) + ' LR: ' + str(
            FLAGS.learning_rate) + ' results in test error: ' + str.format("{0:.5f}", np.sqrt(new_error)))
    else:
        f = open(FLAGS.results_file + 'no_rnn', 'a')
        f.write('\nSimple AE For the data from ' + str(FLAGS.data_dir) + ' Width: ' + str(
            FLAGS.network_width) + ' and depth : ' + str(FLAGS.num_hidden_layers) + ' LR: ' + str(
            FLAGS.learning_rate) + ' results in test error: ' + str.format("{0:.5f}", np.sqrt(new_error)))
    f.close()

    return ae

    
def test(ae,input_seq_file_name, max_val, mean_pose, extract_middle_layer=False):
   with ae.session.graph.as_default() as sess:
    sess = ae.session
    chunking_stride = FLAGS.chunking_stride


    #                    GET THE DATA
        
    # get input sequnce
    #print('\nRead a test sequence from the file',input_seq_file_name,'...')
    original_input = read_test_seq_from_binary(input_seq_file_name)
    
    #print('Preprocess...')
    coords_minus_mean = original_input - mean_pose[np.newaxis,:]
    eps=1e-15
    coords_normalized =np.divide(coords_minus_mean,max_val[np.newaxis,:]+eps)

    if(coords_normalized.shape[0] < ae.sequence_length):
        mupliplication_factor = (ae.batch_size * ae.sequence_length / coords_normalized.shape[0]) + 1
        # Pad the sequence with itself in order to fill the batch completely
        coords_normalized = np.tile(coords_normalized, mupliplication_factor)
        print("Test sequence was way to short!")

    # Split it into chunks
    all_chunks = np.array([coords_normalized[i:i + ae.sequence_length, :] for i in xrange(0, len(original_input)-ae.sequence_length + 1, chunking_stride)]) # Split sequence into chunks

    original_size = all_chunks.shape[0]

    if (original_size < ae.batch_size):
        mupliplication_factor = int(ae.batch_size / all_chunks.shape[0]) + 1
        # Pad the sequence with itself in order to fill the batch completely
        all_chunks = np.tile(all_chunks, (mupliplication_factor,1,1))

    # Batch those chunks
    batches = np.array([all_chunks[i:i + ae.batch_size, :] for i in xrange(0, len(all_chunks)-ae.batch_size + 1, ae.batch_size)])

    numb_of_batches = batches.shape[0]

    
    #                    RUN THE NETWORK

    output_batches = np.array([])

    # Go over all batches one by one
    for batch_numb in range(numb_of_batches):
        output_batch, mask = sess.run([ae._valid_output, ae._mask], feed_dict={ae._valid_input_: batches[batch_numb],
                                                                ae._mask:ae._mask_generator.eval(session=sess)})
        # Take known values into account
        new_result = use_existing_markers(batches[batch_numb], output_batch, mask, FLAGS.defaul_value)#.eval(session=sess)

        output_batches = np.append(output_batches, [new_result],axis=0) if output_batches.size else np.array([new_result])

    #print('Postprocess...')
    output_sequence = reshape_from_batch_to_sequence(output_batches)

    if(extract_middle_layer):
      return output_sequence

    reconstructed = convert_back_to_3d_coords(output_sequence, max_val, mean_pose)

    #              CALCULATE the error for our network
    new_size = np.fmin(reconstructed.shape[0],original_input.shape[0])
    error = (reconstructed[0:new_size] - original_input[0:new_size]) * ae.scaling_factor
    rmse = np.sqrt(((error[error>0.000000001]) ** 2).mean()) # take into account only missing markers

    output_bvh_file_name = FLAGS.data_dir+'/result.txt'
    np.savetxt(output_bvh_file_name, reconstructed, fmt='%.5f', delimiter=' ')
    #print('And write an output into the file ' + output_bvh_file_name + '...')

    return rmse


def reshape_from_batch_to_sequence(input_batch):
    '''
    Reshape batch of overlapping sequences into 1 sequence
    :param input_batch: batch of overlapping sequences
    :return: flat_sequence: one sequence with the same values

    '''

    # Get the data from the Flags
    chunking_stride = FLAGS.chunking_stride
    sequence_length = FLAGS.chunk_length

    # Reshape batches
    input_chunks = input_batch.reshape(-1, input_batch.shape[2], input_batch.shape[3])
    numb_of_chunks = input_chunks.shape[0]

    # Map from overlapping windows to non-overlaping
    # Take first chunk as a whole and the last part of each other chunk
    input_non_overlaping = input_chunks[0]
    for i in range(1, numb_of_chunks, 1):
        input_non_overlaping = np.concatenate(
            (input_non_overlaping, input_chunks[i][sequence_length - chunking_stride: sequence_length][:]),
            axis=0)
    input_non_overlaping = np.array(input_non_overlaping)

    # Flaten it into a sequence
    flat_sequence = input_non_overlaping.reshape(-1, input_non_overlaping.shape[-1])

    return flat_sequence


def convert_back_to_3d_coords(sequence, max_val, mean_pose):
    '''
    Convert back from overlapping batches of the normalized values to original 3d coordinates

    :param sequence: sequence of the normalized values
    :param max_val: maximal value in the dataset
    :param mean_pose: mean value in the dataset

    :return: 3d coordinates corresponding to the batch
    '''

    # Convert it back from the [-1,1] to original values
    reconstructed = np.multiply(sequence, max_val[np.newaxis, :] + 1e-15)
    
    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis,:]

    #Unroll batches into the sequence
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])
    
    return reconstructed

def get_the_data():

  start_time = time.time()
  
  # Get the data
  
  data, max_val,mean_pose = read_all_the_data()
    
  # Check, if we have enough data
  if(FLAGS.batch_size > data.train._num_chunks):
    print('ERROR! Cannot have less train sequences than a batch size!')
    exit(1)
  if(FLAGS.batch_size > data.test._num_chunks):
    print('ERROR! Cannot have less test sequences than a batch size!')
    exit(1)

  return data, max_val,mean_pose

def ignore_right_hand(input_position):
  """ Reduce all the vectore to the one without right_hand

  Args:
    input_position: full body position
  Returns:
    position_wo_r_hand : position, where right hand is ignored and dimension is reduced
  """
  
  coords_before_right_arm = input_position[:, 0 : 24]
  coords_after_right_arm = input_position[:, 36 : 72]
  position_wo_r_hand = np.concatenate((coords_before_right_arm, coords_after_right_arm), axis=1)

  return position_wo_r_hand

if __name__ == '__main__':

  print('DID YOU CHANGED THE TEST_FILE ?')
 
  restore = False # TODO: bring to flags
  learning_rate = FLAGS.learning_rate
  batch_size = FLAGS.batch_size
  dropout = FLAGS.dropout # (keep probability) value

  # Read the data
  data, max_val, mean_pose = get_the_data()

  # Pad max values and the mean pose, if neeeded
  if(FLAGS.amount_of_frames_as_input > 1):
      max_val = np.tile(max_val,FLAGS.amount_of_frames_as_input)
      mean_pose = np.tile(mean_pose, FLAGS.amount_of_frames_as_input)

  # Create the dataset
  '''dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
  print(dataset1.output_types)  # ==> "tf.float32"
  print(dataset1.output_shapes)  # ==> "(10,)"'''

  # Train the network
  ae = learning(restore, data, max_val, learning_rate, batch_size, dropout)

  # TEST it
  rmse = test(ae, FLAGS.data_dir + '/../test_seq/boxing.binary', max_val, mean_pose)
  print("\nOur RMSE for boxing is : ", rmse)

  rmse = test(ae, FLAGS.data_dir + '/../test_seq/basketball.binary', max_val, mean_pose)
  print("\nOur RMSE for basketball is : ", rmse)

  rmse = test(ae, FLAGS.data_dir + '/../test_seq/salto.binary', max_val, mean_pose)
  print("\nOur RMSE for the jump turn is : ", rmse)

  # Close Tf session
  ae.session.close()
