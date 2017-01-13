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

  Autoencoder used for full training cycle, including
  unsupervised pretraining layers and final fine tuning.
  The user specifies the structure of the neural net
  by specifying number of inputs, the number of hidden
  units for each layer and the number of final output
  logits.
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

  def get_variables_to_init(self, n):
    """Return variables that need initialization

    This method aides in the initialization of variables
    before training begins at step n. The returned
    list should be than used as the input to
    tf.initialize_variables

    Args:
      n: int giving step of training
    """
    assert n > 0
    assert n <= self.__num_hidden_layers + 1

    vars_to_init = [self._w(n), self._b(n)]

    if n <= self.__num_hidden_layers:
      vars_to_init.append(self._b(n, "_out"))

    return vars_to_init

  @staticmethod
  def _activate(x, w, b, transpose_w=False):
    y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def pretrain_net(self, numb_layers, input_pl, dropout):
      """Get the output of the autoencoder

      Args:
        numb_layers:    untill which layer should network run
        input_pl: tf placeholder for ae input data
        dropout:  how much of the input neurons will be activated, value in [0,1]
      Returns:
        Tensor of output after 'numb_layers' layers
      """

      assert numb_layers <= self.__num_hidden_layers+1
      #First - Apply Dropout
      last_output = tf.nn.dropout(input_pl, dropout)

      # Then pass through the network
      for i in xrange(numb_layers):
        w = self._w(i + 1)
        b = self._b(i + 1)

        last_output = self._activate(last_output, w, b)

      return last_output


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
      l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(net_output_tf, target_tf)),
                                    reduction_indices=1)) # do we need a square root here?
      return tf.reduce_mean(l2diff,reduction_indices=0)

def main_unsupervised():
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
                                shape=(None, ae_shape[0]),
                                name='ae_input_pl')
        target_ = tf.placeholder(dtype=tf.float32,
                                 shape=(None, ae_shape[0]),
                                 name='ae_target_pl')
        
        output = ae.pretrain_net(ae.num_hidden_layers+1, input_, dropout)

        with tf.name_scope("loss"):
          loss = loss_reconstruction(output, target_)


        # create an optimizer
        optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Get the data
        data = read_unlabeled_data(FLAGS.data_dir, FLAGS.amount_of_subfolders)
        num_train_seq = data.train.num_sequences

        reading_time = (time.time() - start_time)/ 60 # in minutes, instead of seconds

        # Get variables for saving
        variables_to_save = []
        for layer in range(amount_of_layers):
          variables_to_save.append(ae.get_variables_to_init(layer+1))

        # Prepare for making a summary for TensorBoard

        test_error =tf.placeholder(dtype=tf.float32, shape=(), name='eval_error')

        loss_summary = tf.summary.scalar('Train_reconstruction_error', loss)
        step_summary = tf.summary.scalar('Train_global_step', global_step)
        train_summary_op = tf.summary.merge([loss_summary, step_summary])
        test_summary_op =  tf.summary.scalar('Test_reconstr_error',test_error)

        tr_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_train')
        tr_summary_writer = tf.summary.FileWriter(tr_summary_dir)
        test_summary_dir = pjoin(FLAGS.summary_dir, 'last_layer_test')
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)

        test_output = ae.pretrain_net(ae.num_hidden_layers+1, input_, 1) # we do not have dropout during testing

        with tf.name_scope("eval"):
          test_loss = loss_reconstruction(output, target_)
          
        # Pre - Train the whole network jointly
        print("\n\n")
        print("| Tr. Sequence | Error    |   Epoch  |")
        print("|--------------|----------|----------|")

        for step in xrange(FLAGS.pretraining_epochs * num_train_seq):
          feed_dict = fill_feed_dict_ae(data.train, input_, target_, keep_prob, variance, dropout)

          loss_summary, loss_value = sess.run([train_op, loss],
                                              feed_dict=feed_dict)

          if(step%50 == 0):
            # Write summary
            train_summary = sess.run(train_summary_op, feed_dict=feed_dict)
            tr_summary_writer.add_summary(train_summary, step)
              
            # Print results of screen
            output = "| {0:>12} | {1:8.4f} | Epoch {2}  |"\
                       .format(step, loss_value, step // num_train_seq + 1)

            print(output)

            #Evaluate on the test sequences
            error_sum=0
            num_test_seq = data.test.num_sequences
            for test_seq in range(num_test_seq):
              feed_dict = fill_feed_dict_ae(data.test, input_, target_, keep_prob, 0, 1, add_noise=False)
              curr_err = sess.run(test_loss, feed_dict=feed_dict)
              error_sum+= curr_err
              test_error_ = error_sum/num_test_seq
            test_sum = sess.run(test_summary_op, feed_dict={test_error: test_error_})
            test_summary_writer.add_summary(test_sum, step)

    print("Final train error was %.3f, while evarage test error - %.3f." % ( loss_value, test_error_))

    # Create a saver
    saver = tf.train.Saver(np.concatenate(variables_to_save, axis=0).tolist())

    # Save a model
    saver.save(sess,FLAGS.params_file) #TODO : do we need it?

    duration = (time.time() - start_time)/ 60 # in minutes, instead of seconds

    print("The program was running for %.3f  min with %.3f min for reading" % (duration, reading_time))
    #print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
  
  return ae
    

def write_middle_layer(ae, input_seq_file_name, output_seq_file_name):
  print('Extracting middle layer for a test sequence...')
  with ae.session.graph.as_default():
    sess = ae.session
    
    # get input sequnce
    currSequence = read_file(input_seq_file_name)

    # define tensors
    input_ = tf.placeholder(dtype=tf.float32,
                                  shape=(None, FLAGS.DoF),
                                  name='ae_input_pl')

    middle_op = ae.pretrain_net(FLAGS.middle_layer, input_ , 1)
    
    # for each snippet in a sequence
    # pass through the network untill the middle layer
    middle = sess.run(middle_op, feed_dict={input_: currSequence})
    
    # save it into a file
    name = 'Boxing'
    sio.savemat(output_seq_file_name, {'trialId':name, 'spikes':np.transpose(middle)})    
  


if __name__ == '__main__':
  ae = main_unsupervised()
  # get middle layers for visualization
  write_middle_layer(ae, FLAGS.data_dir+'/14/14_01.bvh', FLAGS.data_dir+'/Boxing_middle_layer.txt') 
  # main_supervised(ae)
