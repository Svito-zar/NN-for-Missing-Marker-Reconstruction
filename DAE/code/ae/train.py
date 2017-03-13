from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin
#import datatime

import numpy as np
import scipy.io as sio
import tensorflow as tf
import time
from utils.data import fill_feed_dict_ae, read_unlabeled_data, read_file
from utils.flags import FLAGS
from utils.utils import tile_raster_images
# import class for both architectures of AE
from FlatAE import FlatAutoEncoder    
from HierarchicalAE import HierarchicalAE


def loss_reconstruction(output, target):
  """ Reconstruction error

  Args:
    output: tensor of net output
    target: tensor of net we are trying to reconstruct
  Returns:
    Scalar tensor of mean eucledean distance
  """
  with tf.name_scope("reconstruction_loss"):
      net_output_tf = tf.convert_to_tensor(output, name='input')
      target_tf = tf.convert_to_tensor(target, name='target')
      # Euclidean distance between net_output_tf,target_tf
      l2diff =  tf.nn.l2_loss(tf.subtract(net_output_tf, target_tf))
      return l2diff

def main_unsupervised(restore, pretrain):
  """ Unsupervised pretraining of the autoencoder

  Returns:
    Autoencoder trained on a data provided by FLAGS
  """
  with tf.Graph().as_default() as g:
    sess = tf.Session()

    start_time = time.time()

    # Here is a switch for different AE
    if(FLAGS.Hierarchical):
    
      # Read Hierarchical AE characteristings from flags file
      encode1 = [FLAGS.chest_head_neurons, FLAGS.right_arm_neurons, FLAGS.left_arm_neurons, FLAGS.right_leg_neurons, FLAGS.left_leg_neurons]
      encode2 = [FLAGS.upper_body_neurons, FLAGS.lower_body_neurons]
      encode3 = int(FLAGS.representation_size)

      # Create an autoencoder
      ae = HierarchicalAE(FLAGS.DoF, encode1, encode2, encode3, sess)
    
    else:
      # Get variables from flags
      num_hidden = FLAGS.num_hidden_layers
      ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                          for j in xrange(num_hidden)]
      
      ae_shape = [FLAGS.DoF] + ae_hidden_shapes + [FLAGS.DoF]

      # Create an autoencoder
      ae  = FlatAutoEncoder(ae_shape, sess)

      print('Flat AE was created : ', ae_shape)

    keep_prob = tf.placeholder(tf.float32) #dropout placeholder
    dropout = FLAGS.dropout # (keep probability) value

    #tr_learning_rate = FLAGS.training_learning_rate

    variance = FLAGS.variance_of_noise


    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    batch_size = FLAGS.batch_size
    
    with tf.variable_scope("Train") as main_scope:

        input_ = tf.placeholder(dtype=tf.float32,
                                shape=(FLAGS.batch_size, FLAGS.DoF),
                                name='ae_input_pl')
        target_ = tf.placeholder(dtype=tf.float32,
                                 shape=(FLAGS.batch_size, FLAGS.DoF),
                                 name='ae_target_pl')
        
        output = ae.run_net(input_, dropout)

        with tf.name_scope("training_loss"):
          loss = loss_reconstruction(output, target_)

        # create an optimizer
        optimizer_joint =  tf.train.RMSPropOptimizer(learning_rate=FLAGS.training_learning_rate) #AdamOptimizer(learning_rate=FLAGS.training_learning_rate) # tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')#
        joint_train_op = optimizer_joint.minimize(loss, global_step=global_step, name='Joint_optimizer')

        # Get the data
        print("\n")
        data, max_val,mean_pose = read_unlabeled_data(FLAGS.data_dir, FLAGS.amount_of_subfolders)

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

        test_output = ae.run_net(input_, 1) # we do not have dropout during testing

        with tf.name_scope("eval"):
          test_loss = loss_reconstruction(test_output, target_)

        num_batches = int(data.train._num_examples/batch_size)

        
	#DEBUG
        if(pretrain):
          with tf.name_scope("Pretrain"):

            print('We pretrain for', FLAGS.pretraining_epochs, ' epochs...')

            
            shallow_output = ae.run_shallow(input_, dropout)

            with tf.name_scope("pretraining_loss"):
              shallow_loss = loss_reconstruction(shallow_output, target_)

            # create an optimizer
            shallow_optimizer =  tf.train.AdamOptimizer(learning_rate=FLAGS.pretraining_learning_rate)
            shallow_trainer = shallow_optimizer.minimize(shallow_loss, global_step=global_step, name='Shalow_adam_optimizer')

            # Initialize variables
            sess.run(tf.global_variables_initializer())
          
            # Pre - Train only the last and the first layers
            print('\n Pretrain Hierarchical input and output layers \n')
            print("| Training steps| Error    |   Epoch  |")
            print("|---------------|----------|----------|")

            pretrain_summary_writer = tf.summary.FileWriter(tr_summary_dir)
            
            for step in xrange(FLAGS.pretraining_epochs * num_batches):
              feed_dict = fill_feed_dict_ae(data.train, input_, target_, keep_prob, variance, dropout)

              loss_summary, loss_value  = sess.run([shallow_trainer, shallow_loss],
                                                  feed_dict=feed_dict)
              train_error_ = loss_value/batch_size
              
              if(step%100 == 0):
                # Print results of screen
                output = "| {0:>13} | {1:8.4f} | Epoch {2}  |"\
                           .format(step,  train_error_, data.train._epochs_completed + 1)
                print(output)

            #Reset the count for the actual training
            data.train._epochs_completed = 0
            
        else:
          # Initialize variables
          sess.run(tf.global_variables_initializer())

        tf.get_variable_scope().reuse == True
        
        #restore model, if needed
        if(restore):
          new_saver = tf.train.import_meta_graph(FLAGS.model_dir+'/HierAe.meta')
          new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir+'/'))

        # Create a saver
        saver = tf.train.Saver()  # saver = tf.train.Saver(variables_to_save)
          
        # Train the whole network jointly
        print('We train on ', num_batches, ' batches with ', batch_size, ' training examples in each for', FLAGS.training_epochs, ' epochs...')
        
        print("\n\n")
        print("| Training steps| Error    |   Epoch  |")
        print("|---------------|----------|----------|")

        for step in xrange(FLAGS.training_epochs * num_batches):
          feed_dict = fill_feed_dict_ae(data.train, input_, target_, keep_prob, variance, dropout)

          loss_summary, loss_value  = sess.run([joint_train_op, loss],
                                              feed_dict=feed_dict)

          train_error_ = loss_value/batch_size
          
          if(step%100 == 0):
            # Write summary
            train_summary = sess.run(train_summary_op, feed_dict={train_error: train_error_}) # provide a value for a tensor with a train value
            tr_summary_writer.add_summary(train_summary, step)
              
            # Print results of screen
            output = "| {0:>13} | {1:8.4f} | Epoch {2}  |"\
                       .format(step,  train_error_, data.train._epochs_completed + 1)

            print(output)

            #Evaluate on the test sequences
            error_sum=0
            num_test_batches = int(data.test.num_examples/batch_size)
            for test_batch in range(num_test_batches):
              feed_dict = fill_feed_dict_ae(data.test, input_, target_, keep_prob, 0, 1, add_noise=False)
              curr_err, curr_input = sess.run([test_loss, input_], feed_dict=feed_dict)
              error_sum+= curr_err
            test_error_ = error_sum/(num_test_batches*batch_size)
            test_sum = sess.run(test_summary_op, feed_dict={test_error: test_error_})
            test_summary_writer.add_summary(test_sum, step)

          # Checkpoints
          if(step%5000==0 & step>1000):
            
            # Print an output for a specific sequence into a file
            write_bvh_file(ae, FLAGS.data_dir+'/37/37_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconst_back.bvh')
            
            # Saver for the model
            #curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            #saver.save(sess, 'FLAGS.model_dir', global_step=global_step) # `save` method will call `export_meta_graph` implicitly.
            


    print("Final train error was %.3f, while evarage test error - %.3f." % ( train_error_, test_error_))
    
    # Save a model
    
    #saver.save(sess, FLAGS.model_dir+'/HierAe')

    duration = (time.time() - start_time)/ 60 # in minutes, instead of seconds

    print("The program was running for %.3f  min with %.3f min for reading" % (duration, reading_time))

    # Print an output for a specific sequence into a file
    write_bvh_file(ae, FLAGS.data_dir+'/37/37_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_Flat.bvh')
    # Print an output for a specific sequence into a file
    #write_bvh_file(ae, FLAGS.data_dir+'/25/25_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/reconstr_train.bvh')
  
  return ae
      
  
def write_bvh_file(ae, input_seq_file_name, max_val, mean_pose, output_bvh_file_name):
   print('Take a test sequence from the file',input_seq_file_name)
   print('And write an output into the file ' + output_bvh_file_name + '...')
   with ae.session.graph.as_default():
    sess = ae.session

    # define tensors
    input_ = tf.placeholder(dtype=tf.float32,
                                  shape=(None, FLAGS.DoF),
                                  name='ae_input_pl')

    AE_op = ae.run_net(input_ , 1)
        
    # get input sequnce
    inputSequence = read_file(input_seq_file_name)


    # Define the size of current input sequence
    ae.__curr_batch_size = len(inputSequence)

    #print("Input: ", inputSequence.shape)

    # Substract the mean pose
    inputSequence = inputSequence - mean_pose[np.newaxis,:]

    #print("Mean normalized: ", inputSequence.shape)

    # Scales all values in the input_data to be between -1 and 1
    eps=1e-15
    input_data =np.divide(inputSequence,max_val[np.newaxis,:]+eps)

    #print("Network input :", input_data.shape)

    # pass it through the AE
    outputSequence = sess.run(AE_op, feed_dict={input_: input_data})

    #print("Output: ", outputSequence.shape)

    # Convert it back from [-1,1] to original values
    reconstructed = np.multiply(outputSequence,max_val[np.newaxis,:]+eps)

    #print("Muplyply_back : ", reconstructed)
    
    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis,:]

    # Iclude rotations as well
    rotations = [ [0,0,0] for snippet in range(ae.__curr_batch_size)]
    reconstructed = np.concatenate((reconstructed[:,0:3],rotations,reconstructed[:,3:]), axis=1)
    
    np.savetxt(output_bvh_file_name, reconstructed , fmt='%.5f', delimiter=' ') 

if __name__ == '__main__':
  restore = False
  pretrain = True
  ae = main_unsupervised(restore, pretrain)
  ae.write_middle_layer(FLAGS.data_dir+'/37/37_01.bvh', FLAGS.data_dir+'/middle_layer.bvh', 'Name')

  
     
  # get middle layers for visualization
 # aewrite_middle_layer(ae, FLAGS.data_dir+'/14/14_01.bvh', FLAGS.data_dir+'/Boxing_middle_layer.txt', 'Boxing') """
