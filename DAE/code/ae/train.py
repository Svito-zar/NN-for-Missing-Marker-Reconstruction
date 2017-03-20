from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin
#import datatime

import numpy as np
import scipy.io as sio
import tensorflow as tf
import time
from utils.data import fill_feed_dict_ae, read_unlabeled_data, read_file, loss_reconstruction
from utils.flags import FLAGS
from utils.utils import tile_raster_images
# import class for both architectures of AE
from FlatAE import FlatAutoEncoder    
from HierarchicalAE import HierarchicalAE

def main_unsupervised(restore, pretrain):
  """ Unsupervised pretraining of the autoencoder

  Returns:
    Autoencoder trained on a data provided by FLAGS
  """
  with tf.Graph().as_default() as g:
    sess = tf.Session()

    start_time = time.time()

    # Read the flags
    keep_prob = tf.placeholder(tf.float32) #dropout placeholder
    dropout = FLAGS.dropout # (keep probability) value
    learning_rate = FLAGS.pretraining_learning_rate
    variance = FLAGS.variance_of_noise
    batch_size = FLAGS.batch_size
    chunk_length = FLAGS.chunk_length

    # Check if the flags makes sence
    if(learning_rate < 0 or dropout < 0 or variance < 0):
      print('ERROR! Have got negative values in the flags!')
      exit(1)

    if(FLAGS.amount_of_subfolders < 2):
      print('ERROR! We need at least 2 subfolders: one for training and one for testing!')
      exit(1)


    # Here is a switch for different AE
    if(FLAGS.Hierarchical):
    
      # Read Hierarchical AE characteristings from flags file
      encode1 = [FLAGS.chest_head_neurons, FLAGS.right_arm_neurons, FLAGS.left_arm_neurons, FLAGS.right_leg_neurons, FLAGS.left_leg_neurons]
      encode2 = [FLAGS.upper_body_neurons, FLAGS.lower_body_neurons]
      encode3 = int(FLAGS.representation_size)

      # Create an autoencoder
      ae = HierarchicalAE(FLAGS.DoF, np.array(encode1), np.array(encode2), np.array(encode3), sess)
    
    else:
      # Get variables from flags
      num_hidden = FLAGS.num_hidden_layers
      ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                          for j in xrange(num_hidden)]
      
      ae_shape = [FLAGS.DoF] + ae_hidden_shapes + [FLAGS.DoF]

      # Create an autoencoder
      ae  = FlatAutoEncoder(ae_shape, sess)

      print('Flat AE was created : ', ae_shape)
    
    with tf.variable_scope("Train") as main_scope:

        ##############        DEFINE  Optimizer and training OPERATOR      ####################################
        
        # get an optimizer

        train_op = ae._train_op

        # Create a saver
        saver = tf.train.Saver()  # saver = tf.train.Saver(variables_to_save)

        # Get the data
        data, max_val,mean_pose = read_unlabeled_data(FLAGS.data_dir, FLAGS.amount_of_subfolders)
    
        # Check, if we have enough data
        if(batch_size > data.train._num_chunks):
          print('ERROR! Cannot have less train sequences than a batch size!')
          exit(1)
        if(batch_size > data.test._num_chunks):
          print('ERROR! Cannot have less test sequences than a batch size!')
          exit(1)

        #print('Variations: ', data.train.sigma)
        #print('Max values: ', max_val)

        reading_time = (time.time() - start_time)/ 60 # in minutes, instead of seconds

        # Prepare for making a summary for TensorBoard

        train_error =tf.placeholder(dtype=tf.float32, shape=(), name='train_error')
        test_error =tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')
        train_summary_op = tf.summary.scalar('Train_reconstruction_error', train_error)
        test_summary_op =  tf.summary.scalar('Test_reconstr_error',test_error)

        tr_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_train')
        tr_summary_writer = tf.summary.FileWriter(tr_summary_dir, graph=tf.get_default_graph())
        test_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_test')
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)

        num_batches = int(data.train._num_chunks/data.train._batch_size)
        num_test_batches = int(data.test._num_chunks/batch_size)
        
        test_loss = ae._test_loss

        #DEBUG
        if(pretrain):
          with tf.name_scope("Pretrain"):

            print('\nPretrain for', FLAGS.pretraining_epochs, ' epochs...\n')

            
            shallow_output = ae.process_sequences_shallow(ae._input_, dropout)

            with tf.name_scope("pretraining_loss"):
              shallow_loss = loss_reconstruction(shallow_output, ae._target_)/(batch_size*chunk_length)

            # create an optimizer
            shallow_optimizer =  tf.train.RMSPropOptimizer(learning_rate=FLAGS.pretraining_learning_rate)
            shallow_trainer = shallow_optimizer.minimize(shallow_loss, global_step=tf.contrib.framework.get_or_create_global_step(), name='Shalow_optimizer')

            # Initialize variables
            sess.run(tf.global_variables_initializer())
          
            # Pre - Train only the last and the first layers
            print("| Training steps| Error    |   Epoch  |")
            print("|---------------|----------|----------|")

            pretrain_summary_writer = tf.summary.FileWriter(tr_summary_dir)
            
            for step in xrange(FLAGS.pretraining_epochs * num_batches):
              feed_dict = fill_feed_dict_ae(data.train, ae._input_, ae._target_, keep_prob, variance, dropout)

              loss_summary, loss_value  = sess.run([shallow_trainer, shallow_loss],feed_dict=feed_dict)
              
              if(step%2000 == 0):
                # Print results of screen
                output = "| {0:>13} | {1:8.4f} | Epoch {2}  |"\
                           .format(step,  loss_value, data.train._epochs_completed + 1)
                print(output)

            #Reset the count for the actual training
            data.train._epochs_completed = 0
            
        else:
          # Initialize variables
          sess.run(tf.global_variables_initializer())
        
        #restore model, if needed
        if(restore):
          new_saver = tf.train.import_meta_graph(FLAGS.model_dir+'/HierAe.meta')
          new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir+'/'))
          
        # Train the whole network jointly
        print('\nWe train on ', num_batches, ' batches with ', batch_size, ' training examples in each for', FLAGS.training_epochs, ' epochs...')
        
        print("")
        print("|  Epoch  | Error   |")
        print("|-------- |---------|")


        for epoch in xrange(FLAGS.training_epochs):
          for batches in xrange(num_batches):
              
            feed_dict = fill_feed_dict_ae(data.train, ae._input_, ae._target_, keep_prob, variance, dropout)

            loss_summary, loss_value  = sess.run([ae._train_op, ae._loss],
                                                feed_dict=feed_dict)
            train_error_ = loss_value
                                               
          # Write summary
          train_summary = sess.run(train_summary_op, feed_dict={train_error: train_error_}) # provide a value for a tensor with a train value
          tr_summary_writer.add_summary(train_summary, epoch)
                
          """# Print results of screen
          output = "| Epoch {0:2}|{1:8.4f} |"\
                         .format(data.train._epochs_completed + 1,  train_error_)
          print(output)"""

          #Evaluate on the test sequences
          error_sum=0
          for test_batch in range(num_test_batches):
             feed_dict = fill_feed_dict_ae(data.test, ae._input_, ae._target_, keep_prob, 0, 1, add_noise=False)
             curr_err = sess.run([test_loss], feed_dict=feed_dict)
             error_sum+= curr_err[0]
          test_error_ = error_sum/(num_test_batches)
          test_sum = sess.run(test_summary_op, feed_dict={test_error: test_error_})
          test_summary_writer.add_summary(test_sum, epoch)

          # Print results of screen
          output = "| Epoch {0:2}|{1:8.4f} |"\
                         .format(data.train._epochs_completed + 1,  test_error_)
          print(output)

          # Checkpoints
          if(epoch%20==0 & epoch>0):
              
            # Print an output for a specific sequence into a file
            write_bvh_file(ae, FLAGS.data_dir+'/34/34_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconst_back.bvh')
                
            # Saver for the model
            #curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            saver.save(sess, 'FLAGS.model_dir'+'/FlatAe', global_step=epoch) # `save` method will call `export_meta_graph` implicitly.
    print("\nFinal train error was %.3f, while evarage test error - %.3f." % ( train_error_, test_error_))

    duration = (time.time() - start_time)/ 60 # in minutes, instead of seconds

    print("The training was running for %.3f  min with %.3f min for reading" % (duration, reading_time))

    # Print an output for a specific sequence into a file
    read_process_write_bvh_file(ae, FLAGS.data_dir+'/34/34_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_Hier.bvh')
    # Print an output for a specific sequence into a file
    #write_bvh_file(ae, FLAGS.data_dir+'/25/25_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_train.bvh')
  
  return ae 

def read_process_write_bvh_file(ae,input_seq_file_name, max_val, mean_pose, output_bvh_file_name):
   print('\nTake a test sequence from the file',input_seq_file_name)
   with ae.session.graph.as_default():
    sess = ae.session

    #                    GET THE DATA
        
    # get input sequnce
    inputSequence = read_file(input_seq_file_name)

    # Split it into chunks
    chunks = np.array([inputSequence [i:i + ae.sequence_length, :] for i in xrange(0, len(inputSequence )-ae.sequence_length + 1, FLAGS.chunking_stride)]) # Split sequence into chunks

    # Substract the mean pose
    chunks_minus_mean = chunks - mean_pose[np.newaxis,np.newaxis,:]

    # Scales all values in the input_data to be between -1 and 1
    eps=1e-15
    chunks_normalized =np.divide(chunks_minus_mean,max_val[np.newaxis,np.newaxis,:]+eps)

    # Batch those chunks
    batches = np.array([chunks_normalized[i:i + ae.batch_size, :] for i in xrange(0, len(chunks_normalized)-ae.batch_size + 1, FLAGS.chunking_stride)])

    numb_of_batches = batches.shape[0]

    #                    RUN THE NETWORK

    # pass the batches of chunks through the AE
    output_batches= np.array( [ sess.run(ae._test_output , feed_dict={ae._input_: batches[i]}) for i in range(numb_of_batches)])

    # Unroll it to back to the sequence
    output_chunks = output_batches.reshape(-1, output_batches.shape[-1])

    # Convert it back from [-1,1] to original values
    reconstructed = np.multiply(output_chunks,max_val[np.newaxis,np.newaxis,:]+eps)
    
    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis,np.newaxis,:]

    #Unroll batches into the sequence
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])

    numb_of_chunks = reconstructed.shape[0]

    # Include rotations as well
    rotations = np.array( [  [0,0,0] for time_st  in range(numb_of_chunks)] ) #in range(ae.__sequence_length) for snippet
    reconstructed = np.concatenate((reconstructed[:,0:3],rotations,reconstructed[:,3:]), axis=1)
    
    np.savetxt(output_bvh_file_name, reconstructed , fmt='%.5f', delimiter=' ')

   print('And write an output into the file ' + output_bvh_file_name + '...')
   
if __name__ == '__main__':
  restore = False
  pretrain = FLAGS.Pretraining
  ae = main_unsupervised(restore, pretrain)

  #ae.write_middle_layer(FLAGS.data_dir+'/37/37_01.bvh', FLAGS.data_dir+'/middle_layer.bvh', 'Name')
     
  # get middle layers for visualization
 # aewrite_middle_layer(ae, FLAGS.data_dir+'/14/14_01.bvh', FLAGS.data_dir+'/Boxing_middle_layer.txt', 'Boxing') """
