from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import scipy.io as sio
import tensorflow as tf
import time
from utils.data import fill_feed_dict_ae, read_unlabeled_data, read_file
from utils.flags import FLAGS
from utils.eval import loss_supervised, evaluation, do_eval_summary
from utils.utils import tile_raster_images


class AutoEncoder(object):
  """Generic deep autoencoder.

  Autoencoder used for full training cycle.
  The user specifies the structure of the neural net
  by specifying number of inputs, the number of hidden
  units for each layer and the number of final outputs.
  """
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  def __init__(self, shape, sess):
    """Autoencoder initializer

    Args:
      shape: list of ints specifying
              num input, hidden1 units,...hidden_n units, num logits
      sess: tensorflow session object to use
    """
    self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
    self.__num_hidden_layers = len(self.__shape) - 2

    self.__variables = {}
    self.__sess = sess

    self._setup_variables()

  @property
  def shape(self):
    return self.__shape

  @property
  def num_hidden_layers(self):
    return self.__num_hidden_layers

  @property
  def session(self):
    return self.__sess

  def __getitem__(self, item):
    """Get autoencoder tf variable

    Returns the specified variable created by this object.
    Names are weights#, biases#, biases#_out, weights#_fixed,
    biases#_fixed.

    Args:
     item: string, variables internal name
    Returns:
     Tensorflow variable
    """
    return self.__variables[item]

  def __setitem__(self, key, value):
    """Store a tensorflow variable

    NOTE: Don't call this explicity. It should
    be used only internally when setting up
    variables.

    Args:
      key: string, name of variable
      value: tensorflow variable
    """
    self.__variables[key] = value

  def _setup_variables(self):
    with tf.name_scope("autoencoder_variables"):
      for i in xrange(self.__num_hidden_layers + 1):
        # Train weights
        name_w = self._weights_str.format(i + 1)
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.mul(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                   name=name_w,
                                   trainable=True)
        # Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

        if i <= self.__num_hidden_layers:

          # Pretraining output training biases
          name_b_out = self._biases_str.format(i + 1) + "_out"
          b_shape = (self.__shape[i],)
          b_init = tf.zeros(b_shape)
          self[name_b_out] = tf.Variable(b_init,
                                         trainable=True,
                                         name=name_b_out)

  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n) + suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n) + suffix]

  def get_variables(self):
    """Return list of all variables of Autoencoder
    """

    vars=[]

    for n in range(self.__num_hidden_layers):
      vars.append(self._w(n+1))
      vars.append(self._b(n+1))

    return vars

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.tanh(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b)) # was sigmoid before
    return y

  def run_net(self, input_pl, dropout, just_middle = False):
      """Get the output of the autoencoder

      Args:
        input_pl:     tf placeholder for ae input data
        dropout:      how much of the input neurons will be activated, value in [0,1]
        just_middle : will indicate if we want to extract only the middle layer of the network
      Returns:
        Tensor of output
      """

      if(~just_middle): # if not middle layer
        numb_layers = self.__num_hidden_layers+1
      else:
        numb_layers = FLAGS.middle_layer
      
      #First - Apply Dropout
      last_output = tf.nn.dropout(input_pl, dropout)

      # Then pass through the network
      for i in xrange(numb_layers):
        w = self._w(i + 1)
        b = self._b(i + 1)

        last_output = self._activate(last_output, w, b)

      return last_output

  def write_middle_layer(self, input_seq_file_name, output_seq_file_name, name):
    """ Writing a middle layer into the matlab file

    Args:
      ae:                     ae, middle layer of which we want to save
      input_seq_file_name:    name of the file with the sequence for which we want to get a middle layer
      output_seq_file_name:   name of the file in which we will write a middle layer of the AE
      name:                   name of the  'trial' for these sequence
    Returns:
      nothing
    """
    print('\nExtracting middle layer for a test sequence...')
    
    sess = self.__sess
      
    # get input sequnce
    currSequence = read_file(input_seq_file_name)

    # define tensors
    input_ = tf.placeholder(dtype=tf.float32,
                                  shape=(None, FLAGS.DoF),
                                  name='ae_input_pl')
    # Define the size of current input sequence
    self.__curr_batch_size = sess.run(tf.shape(input_ )[0], feed_dict={input_ : currSequence})

    # Define on an operator
    middle_op = self.run_net(input_ , 1, just_middle = True) # 1 means that we have no dropout
      
    # for each snippet in a sequence
    # pass through the network untill the middle layer
    middle = sess.run(middle_op, feed_dict={input_: currSequence})
      
    # save it into a file
    sio.savemat(output_seq_file_name, {'trialId':name, 'spikes':np.transpose(middle)})  




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
      l2diff =  tf.nn.l2_loss(tf.sub(net_output_tf, target_tf))
      return l2diff

def main_unsupervised(restore):
  """ Unsupervised pretraining of the autoencoder

  Returns:
    Autoencoder trained on a data provided by FLAGS
  """
  with tf.Graph().as_default() as g:
    sess = tf.Session()

    start_time = time.time()

    # Get variables
    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                        for j in xrange(num_hidden)]
    
    ae_shape = [FLAGS.DoF] + ae_hidden_shapes + [FLAGS.DoF]
    
    keep_prob = tf.placeholder(tf.float32) #dropout placeholder
    dropout = FLAGS.dropout # (keep probability) value

    learning_rate = FLAGS.pretraining_learning_rate

    variance = FLAGS.variance_of_noise

    # Create an autoencoder
    ae = AutoEncoder(ae_shape, sess)

    print('Deep AE was created, with a ', ae_shape)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Create a trainer
    amount_of_layers = len(ae_shape)-1 # amount of layers
    
    with tf.variable_scope("pretrain"):

        input_ = tf.placeholder(dtype=tf.float32,
                                shape=(FLAGS.batch_size, ae_shape[0]),
                                name='ae_input_pl')
        target_ = tf.placeholder(dtype=tf.float32,
                                 shape=(FLAGS.batch_size, ae_shape[0]),
                                 name='ae_target_pl')
        
        output = ae.run_net(input_, dropout)

        with tf.name_scope("loss"):
          loss = loss_reconstruction(output, target_)


        # create an optimizer
        optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step, name='Adam_optimizer')

        # Get variables for saving
        variables_to_save = ae.get_variables()

        for variable in variables_to_save:
          tf.add_to_collection('vars', variable)

        # Create a saver
        saver = tf.train.Saver()  # saver = tf.train.Saver(variables_to_save)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Get the data
        data, max_val,mean_pose = read_unlabeled_data(FLAGS.data_dir, FLAGS.amount_of_subfolders)

        print('The maximum value in the dataset after substracting the mean pose was', max_val)

        reading_time = (time.time() - start_time)/ 60 # in minutes, instead of seconds
        
        #num_train_seq = data.train.num_sequences
        batch_size = FLAGS.batch_size

        # Saver for the model
        #saver.save(sess, 'FLAGS.model_dir'+'/flat_ae', global_step=global_step)
        # `save` method will call `export_meta_graph` implicitly.

        #restore model:
        if(restore):
          new_saver = tf.train.import_meta_graph('my-model.meta')
          new_saver.restore(sess, tf.train.latest_checkpoint('./'))
          all_vars = tf.get_collection('vars')
          # DEBUG
          #for v in all_vars:
          #    v_ = sess.run(v)
          #    print(v_)


        #(np.concatenate(variables_to_save, axis=0).tolist())

        # Restore variables
        #print('I am restoring saved variables...')
        #saver.restore(sess, FLAGS.params_file)

        # print(variables_to_save)

        # Prepare for making a summary for TensorBoard

        train_error =tf.placeholder(dtype=tf.float32, shape=(), name='train_error')
        test_error =tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')

        train_summary_op = tf.summary.scalar('Train_reconstruction_error', train_error)
        test_summary_op =  tf.summary.scalar('Test_reconstr_error',test_error)

        tr_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_train')
        tr_summary_writer = tf.summary.FileWriter(tr_summary_dir)
        test_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_test')
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)

        test_output = ae.run_net(input_, 1) # we do not have dropout during testing

        with tf.name_scope("eval"):
          test_loss = loss_reconstruction(output, target_)
          
        # Pre - Train the whole network jointly
        print("\n\n")
        print("| Training steps| Error    |   Epoch  |")
        print("|---------------|----------|----------|")

        for step in xrange(FLAGS.pretraining_epochs * data.train.num_examples):
          feed_dict = fill_feed_dict_ae(data.train, input_, target_, keep_prob, variance, dropout)

          loss_summary, loss_value, curr_input  = sess.run([train_op, loss, input_],
                                              feed_dict=feed_dict)

          train_error_ = loss_value/batch_size
          
          if(step%300 == 0):
            # Write summary
            train_summary = sess.run(train_summary_op, feed_dict={train_error: train_error_}) # provide a value for a tensor with a train value
            tr_summary_writer.add_summary(train_summary, step)
              
            # Print results of screen
            output = "| {0:>12} | {1:8.4f} | Epoch {2}  |"\
                       .format(step,  train_error_, step//data.train.num_examples + 1)

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

    print("Final train error was %.3f, while evarage test error - %.3f." % ( train_error_, test_error_))
    
    # Save a model
    
    #saver.save(sess,FLAGS.params_file) #TODO : do we need it?

    duration = (time.time() - start_time)/ 60 # in minutes, instead of seconds

    print("The program was running for %.3f  min with %.3f min for reading" % (duration, reading_time))

    # Print an output for a specific sequence into a file
    write_bvh_file(ae, FLAGS.data_dir+'/25/25_01.bvh', max_val, mean_pose,  FLAGS.data_dir+'/boxing_reconstr.bvh')
  
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

    # Substract the mean pose
    inputSequence = inputSequence - mean_pose[np.newaxis,:]

    # Scale it to be between -1 and 1 """
    eps=1e-8
    inputSequence *= 1.0 / (max_val + eps)

    # pass it through the AE
    outputSequence = sess.run(AE_op, feed_dict={input_: inputSequence})

    # Convert it back from [0,1] to original values
    eps=1e-8
    reconstructed = (outputSequence * (max_val  + eps))
    
    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis,:]
    
    np.savetxt(output_bvh_file_name, reconstructed , fmt='%.5f', delimiter=' ') 

if __name__ == '__main__':
  restore = False
  ae = main_unsupervised(restore)
  #ae.write_middle_layer(ae, FLAGS.data_dir+'/14/14_01.bvh', FLAGS.data_dir+'/Boxing_middle_layer.txt', 'Boxing') 

  """with tf.Graph().as_default() as g:
    sess = tf.Session()
    
    # Read AE characteristings from flags file
    encode1 = [FLAGS.chest_neurons, FLAGS.head_neurons, FLAGS.right_arm_neurons, FLAGS.left_arm_neurons, FLAGS.right_leg_neurons, FLAGS.left_leg_neurons]
    encode2 = [FLAGS.upper_body_neurons, FLAGS.lower_body_neurons]
    encode3 = int(FLAGS.representation_size)
    decode_shape = [getattr(FLAGS, "decode_size_{0}".format(j + 1))
                          for j in xrange(FLAGS.num_decoding_layers)]
    decode = [FLAGS.representation_size] + decode_shape + [FLAGS.DoF]

    
   # decode.append(FLAGS.DoF) # that must be the last layer now. TODO: it will change

    #print('First layer structure :', encode1)
    # Create an AE
    Hae = HierarchicalDeepAE(FLAGS.DoF, encode1, encode2, encode3, decode , sess)
  # get middle layers for visualization
    Hae.write_middle_layer( FLAGS.data_dir+'/14/short.bvh', FLAGS.data_dir+'/Boxing_middle_layer.txt', 'Boxing') 

 # write_middle_layer(ae, FLAGS.data_dir+'/14/14_01.bvh', FLAGS.data_dir+'/Boxing_middle_layer.txt', 'Boxing') """
