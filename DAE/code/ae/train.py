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

def learning(data, restore, pretrain, learning_rate, batch_size, dropout,variance, middle_layer_size):
  """ Unsupervised pretraining of the autoencoder

  Returns:
    Autoencoder trained on a data provided by FLAGS
  """
  with tf.Graph().as_default() as g:
    sess = tf.Session()

    # Create an AE
    if(FLAGS.Hierarchical):
      
        # Read Hierarchical AE characteristings from flags file
        encode1 = [FLAGS.chest_head_neurons, FLAGS.right_arm_neurons, FLAGS.left_arm_neurons, FLAGS.right_leg_neurons, FLAGS.left_leg_neurons]
        encode2 = [FLAGS.upper_body_neurons, FLAGS.lower_body_neurons]
        encode3 = middle_layer_size

        # Create an autoencoder
        ae = HierarchicalAE(FLAGS.DoF, np.array(encode1), np.array(encode2), np.array(encode3), sess)
      
    else:
        # Get variables from flags
        num_hidden = FLAGS.num_hidden_layers
        ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                            for j in xrange(num_hidden)]
        #set middle layer size
        ae_hidden_shapes[2] = middle_layer_size
        
        ae_shape = [FLAGS.DoF] + ae_hidden_shapes + [FLAGS.DoF]

        # Create an autoencoder
        ae  = FlatAutoEncoder(ae_shape, sess, learning_rate, batch_size, dropout, variance)
        print('Flat AE was created : ', ae_shape)

    start_time = time.time()

    if(restore and pretrain):
      print('ERROR! You cannot restore and pretrain at the same time! Please, chose one of these options')
      exit(1)

    if(pretrain and FLAGS.Layer_wise_Pretraining):
      print('ERROR! You cannot do 2 pretraining at the same time! Please, chose one of them')
      exit(1)

    # Read the flags
    keep_prob = tf.placeholder(tf.float32) #dropout placeholder
    chunk_length = FLAGS.chunk_length

    # Check if the flags makes sence
    if(dropout < 0 or variance < 0):
      print('ERROR! Have got negative values in the flags!')
      exit(1)
    
    
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
        test_error =tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')
        tf.summary.scalar('Train_error', train_error)
        train_summary_op = tf.summary.merge_all()
        test_summary_op =  tf.summary.scalar('Validation_error',test_error)

        tr_summary_dir = pjoin(FLAGS.summary_dir, 'train')
        tr_summary_writer = tf.summary.FileWriter(tr_summary_dir, graph=tf.get_default_graph())
        test_summary_dir = pjoin(FLAGS.summary_dir, 'validation')
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)

        num_batches = int(data.train._num_chunks/batch_size)
        num_test_batches = int(data.test._num_chunks/batch_size)
        
        test_loss = ae._test_loss

          #new_saver = tf.train.import_meta_graph(FLAGS.chkpt_dir+'.meta')
          #new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.chkpt_dir+'/'))

        #restore model, if needed
        if(restore):
          chkpt_file = FLAGS.chkpt_dir+'/chkpt-360'
          saver.restore(sess, chkpt_file)
          print("Model restored from the file "+str(chkpt_file) + '.')

        if(FLAGS.Layer_wise_Pretraining):
          with tf.name_scope("Layer-wise_pretraining"):

            # create an optimizer
            pretrain_optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)

            # Make an array of the trainers for all the layers
            trainers=[pretrain_optimizer.minimize(loss_reconstruction(ae.run_less_layers(ae._input_, i+1), ae.run_less_layers(ae._input_, i+1, is_target=True)), global_step=tf.contrib.framework.get_or_create_global_step(), name='Layer_wise_optimizer_'+str(i)) for i in xrange(len(ae_shape) - 2)]

            # Initialize all the variables
            sess.run(tf.global_variables_initializer())
            
            for i in xrange(len(ae_shape) - 2):
              n = i + 1
              print('Pretrain layer number ', n,' ... ')
              with tf.variable_scope("layer_{0}".format(n)):

                layer = ae.run_less_layers(ae._input_, n)

                with tf.name_scope("pretraining_loss"):
                  target_for_loss = ae.run_less_layers(ae._input_, n, is_target=True)

                loss =  loss_reconstruction(layer, target_for_loss)
                
                pretrain_trainer = trainers[i] #pretrain_optimizer.minimize(loss, global_step=tf.contrib.framework.get_or_create_global_step(), name='Layer_wise_optimizer')

                ''' if(i==0):
                  # Initialize variables, if we have not loaded them yet
                  sess.run(tf.global_variables_initializer())

                else:
                  sess.run(tf.initialize_variables([pretrain_optimizer.get_slot(loss, name) for name in pretrain_optimizer.get_slot_names()]))'''
                                                    
                #guarantee_initialized_variables(sess)
                #sess.run(tf.initialize_variables([pretrain_optimizer.get_slot(loss, name)
                #                  

                layer_wise_pretrain_summary_writer = tf.summary.FileWriter(tr_summary_dir)
                
                for epoch in xrange(FLAGS.pretraining_epochs):
                  for batches in xrange(num_batches):
                    feed_dict = fill_feed_dict_ae(data.train, ae._input_, ae._target_, keep_prob, variance, dropout)

                    loss_summary, loss_value  = sess.run([pretrain_trainer, loss],feed_dict=feed_dict)

                    
                    
                  train_summary = sess.run(train_summary_op, feed_dict={train_error: 2}) # random constant just for debug
                  tr_summary_writer.add_summary(train_summary, epoch +FLAGS.pretraining_epochs*i)
                  '''
                    # Print results of screen
                    output = "| {0:>13} | {1:8.4f} | Epoch {2}  |"\
                               .format(step,  loss_value, data.train._epochs_completed + 1)
                  print(output)'''

            # Copy the trained weights to the fixed matrices and biases
            ae[ae._weights_str.format(n) + 'fixed'] = ae._w(n)
            ae[ae._biases_str.format(n) + 'fixed'] = ae._b(n)
            
          #Reset the count for the actual training
          data.train._epochs_completed = 0

        #Pretrain last and first layer
        if(pretrain):
          with tf.name_scope("Pretrain_first_and_last_layer"):

            print('\nPretrain for', FLAGS.pretraining_epochs, ' epochs...\n')

            
            shallow_output = ae.process_sequences_shallow(ae._input_, dropout)

            with tf.name_scope("pretraining_loss"):
              shallow_loss = loss_reconstruction(shallow_output, ae._target_)/(batch_size*chunk_length)

            # create an optimizer
            shallow_optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)
            shallow_trainer = shallow_optimizer.minimize(shallow_loss, global_step=tf.contrib.framework.get_or_create_global_step(), name='Shalow_optimizer')

            # Initialize variables, if we have not loaded them yet
            sess.run(tf.global_variables_initializer())
          
            # Pre - Train only the last and the first layers
            '''print("| Training steps| Error    |   Epoch  |")
            print("|---------------|----------|----------|")'''

            pretrain_summary_writer = tf.summary.FileWriter(tr_summary_dir)
            
            for step in xrange(FLAGS.pretraining_epochs * num_batches):
              feed_dict = fill_feed_dict_ae(data.train, ae._input_, ae._target_, keep_prob, variance, dropout)

              loss_summary, loss_value  = sess.run([shallow_trainer, shallow_loss],feed_dict=feed_dict)
              
              '''if(step%5000 == 0):
                # Print results of screen
                output = "| {0:>13} | {1:8.4f} | Epoch {2}  |"\
                           .format(step,  loss_value, data.train._epochs_completed + 1)
                print(output)'''
            
            #Reset the count for the actual training
            data.train._epochs_completed = 0
            
        else:
          # Initialize variables
          sess.run(tf.global_variables_initializer())

          
        # Train the whole network jointly
        print('\nWe train on ', num_batches, ' batches with ', batch_size, ' training examples in each for', FLAGS.training_epochs, ' epochs...')
        '''print("")
        print("|  Epoch  | Error   |")
        print("|-------- |---------|")'''

        # A few initialization for the early stopping
        delta = 0.08 # error tolerance for early stopping
        best_error = 10000
        num_valid_batches = int(data.validation._num_chunks/batch_size)
  

        for epoch in xrange(FLAGS.training_epochs):
          for batches in xrange(num_batches):
              
            feed_dict = fill_feed_dict_ae(data.train, ae._input_, ae._target_, keep_prob, variance, dropout)

            loss_summary, loss_value  = sess.run([train_op, ae._reconstruction_loss],
                                                feed_dict=feed_dict)
            train_error_ = loss_value
                                               
          # Print results of screen
          '''output = "| Epoch {0:2}|{1:8.4f} |"\
                         .format(data.train._epochs_completed + 1,  train_error_)
          print(output)'''

          if(epoch%3==0 and epoch>=30):
            # Write summary
            train_summary = sess.run(train_summary_op, feed_dict={train_error: train_error_}) # provide a value for a tensor with a train value
            tr_summary_writer.add_summary(train_summary, epoch)
            
     	    #Evaluate on the validation sequences
            error_sum=0
            for valid_batch in range(num_valid_batches):
              feed_dict = fill_feed_dict_ae(data.validation, ae._input_, ae._target_, keep_prob, 0, 1, add_noise=False)
              curr_err = sess.run([test_loss], feed_dict=feed_dict)
              error_sum+= curr_err[0]
            new_error = error_sum/(num_valid_batches)
            test_sum = sess.run(test_summary_op, feed_dict={test_error: new_error})
            test_summary_writer.add_summary(test_sum, epoch)

            # Early stopping
            if(epoch%5==0 and FLAGS.Early_stopping):             
              if((new_error - best_error) / best_error > delta):
                print('After '+str(epoch) + ' epochs the training started over-fitting ')
                break
              if(new_error < best_error):
                best_error = new_error

                # Saver for the model
                # curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                save_path = saver.save(sess, FLAGS.chkpt_dir+'/chkpt', global_step=epoch) # `save` method will call `export_meta_graph` implicitly.  '''
                print("Model saved in file: %s" % save_path)
                

    # EValuate train error
    error_sum=0
    for train_batch in range(num_batches):
        feed_dict = fill_feed_dict_ae(data.train, ae._input_, ae._target_, keep_prob, 0, 1, add_noise=False)
        curr_err = sess.run([test_loss], feed_dict=feed_dict)
        error_sum+= curr_err[0]
    train_error_ = error_sum/(num_batches)

    # Evaluate final test error
    error_sum=0
    for test_batch in range(num_test_batches):
      feed_dict = fill_feed_dict_ae(data.test, ae._input_, ae._target_, keep_prob, 0, 1, add_noise=False)
      curr_err = sess.run([test_loss], feed_dict=feed_dict)
      error_sum+= curr_err[0]
    test_error_ = error_sum/(num_test_batches)

    duration = (time.time() - start_time)/ 60 # in minutes, instead of seconds

    print("The training was running for %.3f  min" % (duration))

    # Print an output for a specific sequence into a file
    #read_process_write_bvh_file(ae, FLAGS.data_dir+'/dev/16_02.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_eval_box.bvh')
    #if(evaluate):
    #read_process_write_bvh_file(ae, FLAGS.data_dir+'/eval/14_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_flat_no_Recurrency_test.bvh')
    #read_process_write_bvh_file(ae, FLAGS.data_dir+'/02/05_10.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_recurrent_with_64_T_4_layer_train.bvh')
  
  return train_error_, test_error_

def read_process_write_bvh_file(ae,input_seq_file_name, max_val, mean_pose, output_bvh_file_name):
   with ae.session.graph.as_default():
    sess = ae.session
    chunking_stride = FLAGS.chunking_stride

    #                    GET THE DATA
        
    # get input sequnce
    print('\nRead a test sequence from the file',input_seq_file_name,'...')
    inputSequence = read_file(input_seq_file_name, True)

    global_coordinates = inputSequence[:,:6]
    print(global_coordinates.shape)
    local_coordinates = inputSequence[:,6:]
    print(local_coordinates.shape)

    # Split it into chunks
    print('Preprocess...')
    chunks = np.array([local_coordinates[i:i + ae.sequence_length, :] for i in xrange(0, len(inputSequence )-ae.sequence_length + 1, chunking_stride)]) # Split sequence into chunks

    #print(chunks.shape[0], ' chunks')

    # Substract the mean pose
    chunks_minus_mean = chunks - mean_pose[np.newaxis,np.newaxis,:]

    # Scales all values in the input_data to be between -1 and 1
    eps=1e-15
    chunks_normalized =np.divide(chunks_minus_mean,max_val[np.newaxis,np.newaxis,:]+eps)

    #print(chunks_normalized.shape[0], ' chunks after normalization')

    # Batch those chunks
    batches = np.array([chunks_normalized[i:i + ae.batch_size, :] for i in xrange(0, len(chunks_normalized)-ae.batch_size + 1, ae.batch_size)])

    numb_of_batches = batches.shape[0]

    #print(numb_of_batches, ' batches')

    #                    RUN THE NETWORK

    # pass the batches of chunks through the AE
    print('Run the network...')
    output_batches=np.array( [ sess.run(ae._test_output , feed_dict={ae._input_: batches[i]}) for i in range(numb_of_batches)])

    #print(output_batches.shape[0], ' output batches')
    
    # Unroll it to back to the sequence
    print('Postprocess...')
    #output_chunks = output_batches.reshape(-1, output_batches.shape[-1])

    output_chunks = output_batches.reshape(-1, output_batches.shape[2], output_batches.shape[3])

    numb_of_chunks = output_chunks.shape[0]

    #print(numb_of_chunks, ' output chunks')

    # Map from overlapping windows to non-overlaping
    # Take first chunk as a whole and the last part of each other chunk
    output_non_overlaping = output_chunks[0]
    for i in range(1,numb_of_chunks,1):
      output_non_overlaping = np.concatenate((output_non_overlaping, output_chunks[i][ae.sequence_length-chunking_stride:ae.sequence_length][:] ), axis=0)
    output_non_overlaping = np.array(output_non_overlaping)

    # Flaten it into a sequence
    output_sequence = output_non_overlaping.reshape(-1, output_non_overlaping.shape[-1])

    # Convert it back from [-1,1] to original values
    reconstructed = np.multiply(output_non_overlaping,max_val[np.newaxis,np.newaxis,:]+eps)
    
    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis,np.newaxis,:]

    #Unroll batches into the sequence
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])
    numb_of_frames = reconstructed.shape[0]

    # Try to nulify global coordinates
    global_coordinates= [[0 for i in range(FLAGS.global_DoF)] for j in range(numb_of_frames)]

    # Include global coordinates as well
    reconstructed = np.concatenate((global_coordinates[0:numb_of_frames],reconstructed), axis=1)
    
    np.savetxt(output_bvh_file_name, reconstructed , fmt='%.5f', delimiter=' ')

   print('And write an output into the file ' + output_bvh_file_name + '...')

def get_the_data(evaluate):

  start_time = time.time()
  
  # Get the data
  data, max_val,mean_pose = read_unlabeled_data(FLAGS.data_dir, FLAGS.amount_of_subfolders, evaluate)
    
  # Check, if we have enough data
  if(FLAGS.batch_size > data.train._num_chunks):
    print('ERROR! Cannot have less train sequences than a batch size!')
    exit(1)
  if(FLAGS.batch_size > data.test._num_chunks):
    print('ERROR! Cannot have less test sequences than a batch size!')
    exit(1)

  reading_time = (time.time() - start_time)/ 60 # in minutes, instead of seconds

  print('Reading of data took ' + str(reading_time) + ' minutes')

  return data, max_val,mean_pose
   
if __name__ == '__main__':
 
  restore = False
  pretrain = FLAGS.Pretraining
  learning_rate = FLAGS.training_learning_rate
  batch_size = FLAGS.batch_size
  dropout = FLAGS.dropout # (keep probability) value
  variance = FLAGS.variance_of_noise
  middle_layer_size = FLAGS.representation_size
  weight_decay = FLAGS.Weight_decay
  
  print('Fixed hyper-parameters:\n')

  print('learning_rate : ' + str(learning_rate))
  print('batch_size: '+ str(batch_size))
  print('dropout: ' + str(dropout))
  print('variance of noise added to the data: ' + str(variance))
  print('middle layer size: ' + str(middle_layer_size))
  print('Weight decay: ' + str(weight_decay))
  
  evaluate=True

  if(evaluate):
    data, max_val,mean_pose = get_the_data(evaluate)
    train_err, test_err = learning(data, restore, pretrain, learning_rate, batch_size, dropout,variance, middle_layer_size)
    print('For the learning rate ' + str(learning_rate)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))
 
  else:
    data, max_val,mean_pose = get_the_data(evaluate)  
    print('\nWe optimize : pretraining learning rate\n')
    initial_lr = 0.0000125
    for lr_factor in np.logspace(0,8, num=9, base=2):
      lr = lr_factor*initial_lr
      train_err, test_err = learning(data, restore, pretrain, lr, batch_size, dropout,variance, middle_layer_size)
      print('For the learning rate ' + str(lr)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))
   
  
  '''
  #debug
  evaluate = False
  data, max_val,mean_pose = get_the_data(evaluate)
  train_err, test_err = learning(data, restore, pretrain, learning_rate, batch_size, dropout,variance, middle_layer_size)
  print('With EVALUATE FALSE For the learning rate ' + str(learning_rate)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))'''

   
  '''

  
  evaluate = True
  data, max_val,mean_pose = get_the_data(evaluate)
  print('\nWe optimize : middle layer size\n')  
  for middle_layer in np.linspace(70, 140, 8):
    middle_layer_size = int(middle_layer)
    train_err, test_err = learning(data, restore, pretrain, learning_rate, batch_size, dropout,variance, middle_layer_size)
    print('For the middle layer size ' + str(middle_layer_size)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))

  

  print('\nWe optimize : dropout rate\n')
  for dropout in np.linspace(0.7, 0.9, 5):
    train_err, test_err = learning(data, restore, pretrain, learning_rate, batch_size, dropout,variance, middle_layer_size)
    print('For the droput ' + str(dropout)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))
          
  print('\nWe optimize : variance rate\n')
  for variance in np.linspace(0.1, 0.4, 6):
    train_err, test_err = learning(data, restore, pretrain, learning_rate, batch_size, dropout,variance, middle_layer_size)
    print('For variance ' + str(variance)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))

    
    # Print an output for a specific sequence into a file
    #write_bvh_file(ae, FLAGS.data_dir+'/25/25_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_train.bvh')
 
  #ae.write_middle_layer(FLAGS.data_dir+'/37/37_01.bvh', FLAGS.data_dir+'/middle_layer.bvh', 'Name')
     
  # get middle layers for visualization
 # aewrite_middle_layer(ae, FLAGS.data_dir+'/14/14_01.bvh', FLAGS.data_dir+'/Boxing_middle_layer.txt', 'Boxing') 
  '''

